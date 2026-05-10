OpenAI Codex v0.120.0 (research preview)
--------
workdir: Q:\finance-analyzer\.worktrees\adv-orchestration
model: gpt-5.4
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR, C:\Users\Herc2\.codex\memories]
reasoning effort: xhigh
reasoning summaries: none
session id: 019e127e-5e69-7680-bd4a-b10b34274159
--------
user
You are an adversarial code reviewer cross-critiquing another reviewer's findings.

CONTEXT
- Codebase: Q:\finance-analyzer (autonomous trading system, Python).
- Subsystem: orchestration
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
# Adversarial Review: orchestration subsystem (2026-05-08)

[P0] portfolio/agent_invocation.py:1142
**Timeout-enforcement dead code when `_agent_timeout == 0`.**
Problem: Truthiness check on `_agent_timeout` is falsy at 0, so the timeout branch is skipped and the subprocess runs forever. Silent no-op on timeout.
Fix: Use explicit `if _agent_timeout > 0 and elapsed > _agent_timeout`.

[P0] portfolio/agent_invocation.py:839
**Byte-offset capture for auth-error scan is racy.**
Problem: Offset captured before file open creates a window where early subprocess output is missed by the silent-auth-failure detector — exactly the failure mode of the Mar–Apr 2026 outage.
Fix: Capture offset *after* opening the log file handle.

[P0] portfolio/trigger.py:231
**Consensus baseline consumed even when ranging dampening suppresses trigger.**
Problem: Baseline updated regardless of whether Layer 2 was invoked. Next valid crossing is missed because baseline already shifted.
Fix: Only update baseline when trigger fires; keep stale baseline through dampened cycles.

[P1] portfolio/agent_invocation.py:580
**Module-global `_agent_timeout` stale during reentrancy.**
Problem: New trigger overwrites timeout of an old running agent. T1 trigger arriving while T3 agent runs replaces 900s with 120s, and the still-running T3 is killed prematurely.
Fix: Use `{pid: timeout}` dict keyed by spawned process.

[P1] portfolio/agent_invocation.py:1323
**`_agent_log_start_offset` not cleared on completion.**
Problem: Inconsistent cleanup; stale offset persists into next invocation, distorting the auth-failure scan window.
Fix: Clear offset alongside other globals in the completion block.

[P1] portfolio/agent_invocation.py:544
**Stack-overflow auto-disable is sticky.**
Problem: Counter reaching 5 disables agent spawn permanently — Layer 2 never recovers without manual reset. No decay path.
Fix: Add 24h decay on the failure counter.

[P1] portfolio/trigger.py:189
**Startup grace period flag ignored on in-process loop restart.**
Problem: Module-level grace flag not reset on restart-without-reimport; spurious triggers fire immediately after a soft restart.
Fix: Tie grace to current process PID.

[P1] portfolio/agent_invocation.py:293
**Decision-feedback loop does O(N) full-journal scans per invocation.**
Problem: Journal grows unbounded over weeks (10K+ entries); each invocation linearly scans the whole file. Cumulative drag.
Fix: Cap to last 100 entries, or use journal_index for tail reads.

[P1] portfolio/main.py:658
**`ThreadPoolExecutor.cancel_futures` is best-effort only.**
Problem: Running threads not killed; resources leak (open files, sockets, VRAM if Chronos in-flight).
Fix: Add 5s grace period + log orphans; consider hard-kill for known-runaway tasks.

[P1] portfolio/agent_invocation.py:1148
**Timeout-kill failure not persisted.**
Problem: `kill_ok=False` lost; hung agent indistinguishable from killed agent in `invocations.jsonl`.
Fix: Include `kill_status` field in journal record.

[P1] portfolio/trigger.py:434
**`classify_tier()` uses UTC weekday for market-hours decisions.**
Problem: Wrong day at UTC/CET boundary (e.g., 23:00 UTC Thursday = 00:00 CET Friday). Tier misclassified at the day rollover.
Fix: Convert to CET before computing weekday.

[P1] portfolio/health.py:140
**Heartbeat keepalive daemon killed abruptly on loop exit.**
Problem: 2s grace insufficient for daemon mid-write to `health_state.json`. Corruption risk on graceful shutdown.
Fix: Drain daemon (signal + join with longer timeout) before main thread exits.

[P1] portfolio/agent_invocation.py:809
**Broad `except Exception:` masks `ImportError`.**
Problem: All exceptions swallowed; import failure silently falls through to bat-file invocation, hiding broken Python paths.
Fix: Catch specifically `FileNotFoundError`/`OSError`; re-raise `ImportError` so it surfaces.

## Summary

3 P0 + 10 P1. Themes: silent timeout failures, baseline poisoning under dampening,
stale module-globals across reentrant invocations, UTC vs CET drift, broad except
masking real errors. Several echo prior outage patterns (Mar–Apr silent auth) — the
detector is still racy.
=== END ===

Use git/grep/read tools as needed. Be ruthless and cite code. Do not soften.

codex
I’m verifying each cited finding against the actual orchestration code. First step is to pull the referenced line ranges from the four files and inspect the surrounding control flow before judging Claude’s claims.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\agent_invocation.py'; "'$ranges=@(@{s=260;e=330},@{s=520;e=610},@{s=780;e=860},@{s=1120;e=1165},@{s=1300;e=1345}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\main.py'; "'$ranges=@(@{s=630;e=690}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\trigger.py'; "'$ranges=@(@{s=160;e=260},@{s=410;e=460}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\health.py'; "'$ranges=@(@{s=110;e=170}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 880ms:
  110:     least one heartbeat even if it finishes before the first interval).
  111:     Subsequent beats run on the daemon thread until __exit__.
  112: 
  113:     Failure-tolerant by design: tick exceptions are swallowed at WARNING
  114:     level — a Disk-full or permission-denied during keepalive must never
  115:     abort an in-flight Layer 2 trade decision.
  116:     """
  117: 
  118:     def __init__(self, interval: float = _HEARTBEAT_KEEPALIVE_INTERVAL_S) -> None:
  119:         self._interval = interval
  120:         self._stop = threading.Event()
  121:         self._thread: threading.Thread | None = None
  122: 
  123:     def __enter__(self) -> "heartbeat_keepalive":
  124:         # Synchronous first beat — covers the case where the wrapped call
  125:         # returns before the keepalive thread's first tick.
  126:         try:
  127:             heartbeat()
  128:         except Exception:
  129:             logger.warning("heartbeat_keepalive initial beat failed", exc_info=True)
  130: 
  131:         self._thread = threading.Thread(
  132:             target=self._run, daemon=True, name="heartbeat-keepalive",
  133:         )
  134:         self._thread.start()
  135:         return self
  136: 
  137:     def __exit__(self, *_exc: object) -> None:
  138:         self._stop.set()
  139:         if self._thread is not None:
  140:             self._thread.join(timeout=2.0)
  141: 
  142:     def _run(self) -> None:
  143:         # Event.wait returns True when set (stop signaled), False on timeout.
  144:         # So we tick on each timeout and exit on the first True.
  145:         while not self._stop.wait(self._interval):
  146:             try:
  147:                 heartbeat()
  148:             except Exception:
  149:                 logger.warning("heartbeat_keepalive tick failed", exc_info=True)
  150: 
  151: 
  152: def check_staleness(max_age_seconds: int = 300) -> tuple:
  153:     """Check if the loop heartbeat is stale.
  154:     Returns (is_stale: bool, age_seconds: float, state: dict)
  155:     """
  156:     state = load_health()
  157:     hb = state.get("last_heartbeat")
  158:     if not hb:
  159:         return True, float("inf"), state
  160:     last = datetime.fromisoformat(hb)
  161:     age = (datetime.now(UTC) - last).total_seconds()
  162:     return age > max_age_seconds, age, state
  163: 
  164: 
  165: def check_agent_silence(max_market_seconds: int = 7200,
  166:                         max_offhours_seconds: int = 14400) -> dict:
  167:     """Detect silent Layer 2 agent (no invocation for too long).
  168: 
  169:     Args:
  170:         max_market_seconds: Max allowed silence during market hours (default 2h).

 succeeded in 901ms:
  160: def check_triggers(signals, prices_usd, fear_greeds, sentiments):
  161:     global _startup_grace_active
  162:     state = _load_state()
  163:     state["_current_tickers"] = set(signals.keys())  # for pruning in _save_state
  164: 
  165:     # Startup grace period: on the first iteration after a restart, update the
  166:     # baseline (prices, signals, consensus) WITHOUT triggering Layer 2.
  167:     # This lets the loop restart for code updates without spurious T3 reviews.
  168:     current_pid = os.getpid()
  169:     saved_pid = state.get(_GRACE_PERIOD_KEY)
  170:     if _startup_grace_active and saved_pid != current_pid:
  171:         import logging
  172:         _logger = logging.getLogger("portfolio.trigger")
  173:         _logger.info(
  174:             "Startup grace period: updating baseline without triggering "
  175:             "(pid %s -> %s)", saved_pid, current_pid,
  176:         )
  177:         state[_GRACE_PERIOD_KEY] = current_pid
  178:         # Update baselines so next iteration compares from NOW
  179:         state["last"] = {
  180:             "signals": {
  181:                 t: {"action": s["action"], "confidence": s["confidence"]}
  182:                 for t, s in signals.items()
  183:             },
  184:             "prices": dict(prices_usd),
  185:             "fear_greeds": {
  186:                 t: fg if isinstance(fg, dict) else {} for t, fg in fear_greeds.items()
  187:             },
  188:             "sentiments": dict(sentiments),
  189:             "time": time.time(),
  190:         }
  191:         # Update triggered_consensus baseline to current state
  192:         tc = state.get("triggered_consensus", {})
  193:         for ticker, sig in signals.items():
  194:             tc[ticker] = sig["action"]
  195:         state["triggered_consensus"] = tc
  196:         state["today_date"] = _today_str()
  197:         _startup_grace_active = False
  198:         _save_state(state)
  199:         return False, []
  200: 
  201:     _startup_grace_active = False
  202:     prev = state.get("last", {})
  203:     sustained = state.get("sustained_counts", {})
  204:     reasons = []
  205: 
  206:     # 0. Trade reset — if Layer 2 made a trade, trigger reassessment
  207:     if _check_recent_trade(state):
  208:         state["last_trigger_time"] = 0
  209:         reasons.append("post-trade reassessment")
  210: 
  211:     # 1. Signal consensus — trigger ONLY when a ticker first reaches BUY/SELL
  212:     #    from HOLD. BUY↔SELL direction flips are handled by the sustained flip
  213:     #    trigger (#2). Uses persistent triggered_consensus that is NOT wiped
  214:     #    when unrelated triggers (sentiment, etc.) fire.
  215:     #
  216:     #    Ranging regime dampening (2026-04-22): in ranging regime, low-confidence
  217:     #    consensus crossings are noise — require RANGING_CONSENSUS_MIN_CONFIDENCE
  218:     #    to actually fire the trigger. Prevents 20+ HOLD invocations per day.
  219:     triggered_consensus = state.get("triggered_consensus", {})
  220:     for ticker, sig in signals.items():
  221:         action = sig["action"]
  222:         last_tc = triggered_consensus.get(ticker, "HOLD")
  223:         if action in ("BUY", "SELL") and last_tc == "HOLD":
  224:             conf = sig.get("confidence", 0)
  225:             # Ranging regime dampening: skip low-confidence consensus triggers
  226:             ticker_regime = (sig.get("extra") or {}).get("_regime", "unknown")
  227:             if (
  228:                 ticker_regime == "ranging"
  229:                 and RANGING_CONSENSUS_MIN_CONFIDENCE > 0
  230:                 and conf < RANGING_CONSENSUS_MIN_CONFIDENCE
  231:             ):
  232:                 logger.info(
  233:                     "Ranging dampening: %s consensus %s (%.0f%%) suppressed "
  234:                     "(min %.0f%%)",
  235:                     ticker, action, conf * 100,
  236:                     RANGING_CONSENSUS_MIN_CONFIDENCE * 100,
  237:                 )
  238:                 # Still update baseline so we don't re-trigger next cycle
  239:                 triggered_consensus[ticker] = action
  240:                 continue
  241:             # New consensus from HOLD — trigger
  242:             reasons.append(f"{ticker} consensus {action} ({conf:.0%})")
  243:             triggered_consensus[ticker] = action
  244:         elif action == "HOLD" and last_tc != "HOLD":
  245:             # Consensus cleared — reset so next BUY/SELL is "new"
  246:             triggered_consensus[ticker] = "HOLD"
  247:         elif action in ("BUY", "SELL") and action != last_tc:
  248:             # Direction flip (BUY↔SELL) — update baseline silently,
  249:             # let sustained flip trigger (#2) handle it
  250:             triggered_consensus[ticker] = action
  251:     state["triggered_consensus"] = triggered_consensus
  252: 
  253:     # 2. Signal flip — triggers when the new action has been seen for
  254:     #    SUSTAINED_CHECKS consecutive cycles OR for SUSTAINED_DURATION_S
  255:     #    wall-clock seconds, whichever comes first. The duration gate was
  256:     #    added 2026-04-09 so the trigger still fires within ~1 cycle at
  257:     #    long cadences (e.g. 600s); at the historical 60s cadence the count
  258:     #    gate still dominates and behavior is unchanged.
  259:     prev_triggered = prev.get("signals", {})
  260:     flip_cooldowns = state.get("flip_cooldowns", {})
  410: 
  411: def _should_downshift_to_t1(reasons, threshold: float | None = None) -> bool:
  412:     """Decide whether a T2 tier should be downshifted to T1.
  413: 
  414:     Returns True only when every reason is either a low-conviction consensus
  415:     crossing or a fade flip — i.e. all reasons are individually downshiftable.
  416:     A single high-conviction or non-consensus reason blocks downshift.
  417: 
  418:     Empty reason list returns False (no downshift). Called only after
  419:     classify_tier() has already chosen T2 — T1 and T3 are never affected.
  420: 
  421:     threshold=None (default) looks up TIER_DOWNSHIFT_CONFIDENCE at call time,
  422:     allowing runtime overrides via mock.patch or module-attribute reassignment
  423:     (the module-level constant is the single config knob). Passing an explicit
  424:     float overrides for testing.
  425:     """
  426:     if not reasons:
  427:         return False
  428:     effective = TIER_DOWNSHIFT_CONFIDENCE if threshold is None else threshold
  429:     return all(_reason_is_downshiftable(r, effective) for r in reasons)
  430: 
  431: 
  432: def classify_tier(reasons, state=None):
  433:     """Classify trigger reasons into invocation tier (1=quick, 2=signal, 3=full).
  434: 
  435:     Tier 3 (Full Review): periodic review, F&G extreme, first of day.
  436:     Tier 2 (Signal Analysis): new consensus, price moves, post-trade, signal flips.
  437:     Tier 1 (Quick Check): sentiment noise, repeated triggers.
  438: 
  439:     M10/NEW-4: pass state=<dict> to avoid a redundant disk read when the caller
  440:     already has the trigger state loaded. Falls back to loading from file if None.
  441:     """
  442:     if state is None:
  443:         state = _load_state()
  444: 
  445:     # Tier 3: periodic full review
  446:     last_full = state.get("last_full_review_time", 0)
  447:     hours_since = (time.time() - last_full) / 3600
  448: 
  449:     now_utc = datetime.now(UTC)
  450:     from portfolio.market_timing import _eu_market_open_hour_utc, _market_close_hour_utc
  451:     close_hour = _market_close_hour_utc(now_utc)
  452:     eu_open = _eu_market_open_hour_utc(now_utc)
  453:     market_open = now_utc.weekday() < 5 and eu_open <= now_utc.hour < close_hour
  454: 
  455:     # C4/NEW-2: first-of-day T3 check must precede the off-hours periodic cap.
  456:     # An off-hours trigger 4+ hours after the last full review would otherwise
  457:     # return T1 early (line below), skipping the first-of-day T3 entirely.
  458:     if state.get("last_trigger_date") != _today_str():
  459:         return 3  # first real trigger of the day
  460: 

 succeeded in 956ms:
  630:     except TimeoutError:
  631:         timed_out = [n for f, n in futures.items() if not f.done()]
  632:         try:
  633:             from portfolio.signal_engine import get_last_signal as _get_last
  634:             from portfolio.signal_engine import get_phase_log as _get_phase_log
  635:             last_sigs = {n: _get_last(n) for n in timed_out}
  636:             # 2026-04-15: also dump per-ticker phase breakdown when the pool
  637:             # times out. This tells us WHICH post-dispatch phase
  638:             # (acc_load / utility_overlay / weighted_consensus / penalties /
  639:             # linear_factor / consensus_gate / regime_gate) burned the time,
  640:             # so we can target the real bottleneck instead of coarsely blaming
  641:             # __post_dispatch__.
  642:             phase_logs = {n: _get_phase_log(n) for n in timed_out}
  643:         except Exception:
  644:             last_sigs = {}
  645:             phase_logs = {}
  646:         logger.error(
  647:             "BUG-178: Ticker pool timeout after %ds. Stuck: %s. Last signals: %s",
  648:             _TICKER_POOL_TIMEOUT, timed_out, last_sigs,
  649:         )
  650:         for name, phases in phase_logs.items():
  651:             if phases:
  652:                 # Format as 'phase=dur_s' pairs, one ticker per line. Keep on
  653:                 # one log line per ticker so Windows Event Log / tail -f stays
  654:                 # readable when 5 tickers time out simultaneously.
  655:                 phase_str = " ".join(f"{p}={d:.1f}s" for p, d in phases)
  656:                 logger.error("BUG-178 phases [%s]: %s", name, phase_str)
  657:         for f in futures:
  658:             f.cancel()
  659:         signals_failed += len(timed_out)
  660:     finally:
  661:         pool.shutdown(wait=False, cancel_futures=True)
  662: 
  663:     # --- Post-cycle LLM batch flush ---
  664:     # Ministral/Qwen3/fingpt cache misses were enqueued during parallel
  665:     # ticker processing. Now flush them sequentially, grouped by model
  666:     # (max 2 swaps: ministral → qwen3 → fingpt). Fingpt phase added
  667:     # 2026-04-09 as part of feat/fingpt-in-llmbatch which retired the
  668:     # bespoke scripts/fingpt_daemon.py. The sentiment A/B log write is
  669:     # also deferred: flush_ab_log() below walks sentiment._pending_ab_entries
  670:     # and assembles the final rows once Phase 3 has stashed fingpt results.
  671:     try:
  672:         from portfolio.llm_batch import _lock as _llm_lock
  673:         from portfolio.llm_batch import _ministral_queue, _qwen3_queue, flush_llm_batch
  674:         from portfolio.shared_state import MINISTRAL_TTL, _update_cache
  675:         # H24/SS2: Capture queued keys before flush to clear stuck loading keys.
  676:         with _llm_lock:
  677:             _queued_keys = {k for k, _ in _ministral_queue} | {k for k, _ in _qwen3_queue}
  678:         batch_results = flush_llm_batch()
  679:         for cache_key, result in batch_results.items():
  680:             _update_cache(cache_key, result, ttl=MINISTRAL_TTL)
  681:         # Clear loading keys for items that didn't return results (retry next cycle).
  682:         for key in _queued_keys:
  683:             if key not in batch_results:
  684:                 _update_cache(key, None, ttl=60)
  685:         # Now that Phase 3 (fingpt) has stashed its results into
  686:         # sentiment._pending_ab_entries via _stash_fingpt_result, write out
  687:         # the sentiment_ab_log.jsonl rows for this cycle. Must run AFTER
  688:         # flush_llm_batch so the fingpt shadow data is available.
  689:         from portfolio.sentiment import flush_ab_log
  690:         flush_ab_log()

 succeeded in 983ms:
  260:     with what Layer 2 sees in its prompt context, and is much cheaper.
  261: 
  262:     Returns a dict shaped like trade_guards.get_all_guard_warnings():
  263:         {"warnings": [...], "summary": "..."}
  264:     Defaults to empty/no_summary when agent_summary is missing or has
  265:     no trade_guard_warnings field — caller treats that as "no blocks".
  266:     """
  267:     from portfolio.file_utils import load_json
  268:     summary_path = DATA_DIR / "agent_summary.json"
  269:     summary = load_json(summary_path, default=None)
  270:     if not isinstance(summary, dict):
  271:         return {"warnings": [], "summary": "no_summary"}
  272:     guard_block = summary.get("trade_guard_warnings")
  273:     if not isinstance(guard_block, dict):
  274:         return {"warnings": [], "summary": "no_warnings"}
  275:     # Normalize: ensure required keys exist so callers don't have to .get()
  276:     return {
  277:         "warnings": guard_block.get("warnings", []) or [],
  278:         "summary": guard_block.get("summary", ""),
  279:     }
  280: 
  281: 
  282: def _build_decision_feedback(ticker, max_entries=5):
  283:     """Build recent-decision feedback for the trigger ticker.
  284: 
  285:     Scans layer2_journal.jsonl (most-recent-first) for entries mentioning
  286:     *ticker* in the trigger string or the tickers dict.  Returns a formatted
  287:     block that Layer 2 can use to calibrate against its own prior calls, or
  288:     an empty string when no relevant history exists.
  289: 
  290:     Token budget: ≤15 lines.  Never fails the invocation on error.
  291:     """
  292:     try:
  293:         entries = load_jsonl(JOURNAL_FILE)
  294:     except Exception:
  295:         return ""
  296:     if not entries:
  297:         return ""
  298: 
  299:     relevant = []
  300:     for e in reversed(entries):  # most recent first
  301:         trigger = e.get("trigger", "")
  302:         tickers = e.get("tickers", {})
  303:         if ticker in trigger or ticker in tickers:
  304:             relevant.append(e)
  305:             if len(relevant) >= max_entries:
  306:                 break
  307: 
  308:     if not relevant:
  309:         return ""
  310: 
  311:     lines = [f"[RECENT DECISIONS FOR {ticker}]"]
  312:     for e in relevant:
  313:         ts = e.get("ts", "?")[:16]
  314:         decisions = e.get("decisions", {})
  315:         prices = e.get("prices", {})
  316:         price = prices.get(ticker)
  317: 
  318:         parts = []
  319:         for strat, d in decisions.items():
  320:             action = d.get("action", "HOLD")
  321:             parts.append(f"{strat}={action}")
  322:         action_str = ", ".join(parts) if parts else "?"
  323:         price_str = f"${price:,.2f}" if isinstance(price, (int, float)) else "?"
  324:         lines.append(f"  - {ts}: {action_str} @ {price_str}")
  325: 
  326:     lines.append(
  327:         "  Review: were these decisions correct given current price? "
  328:         "Has the thesis changed?"
  329:     )
  330:     return "\n".join(lines)
  520:     # AFTER closing _agent_log (so any buffered output is flushed) but
  521:     # BEFORE _agent_proc / _agent_tier / _agent_reasons are cleared (so
  522:     # the auth-failure record carries the right tier + trigger context).
  523:     # Best-effort: failures are swallowed inside the helper so a busted
  524:     # log read can never break the kill path.
  525:     auth_label = f"layer2_t{_agent_tier}_timeout" if _agent_tier else "layer2_timeout"
  526:     _scan_agent_log_for_auth_failure(auth_label)
  527: 
  528:     # BUG-91: Log the timed-out invocation before returning
  529:     _log_trigger(
  530:         _agent_reasons or fallback_reasons or [],
  531:         "timeout",
  532:         tier=_agent_tier or fallback_tier,
  533:     )
  534: 
  535:     _agent_proc = None
  536:     return kill_ok
  537: 
  538: 
  539: def invoke_agent(reasons, tier=3):
  540:     global _agent_proc, _agent_log, _agent_start, _agent_start_wall, _agent_timeout
  541:     global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before
  542: 
  543:     # Check if Layer 2 is auto-disabled due to consecutive stack overflows
  544:     if _consecutive_stack_overflows >= _MAX_STACK_OVERFLOWS:
  545:         logger.info(
  546:             "Layer 2 skipped: auto-disabled after %d consecutive stack overflows",
  547:             _consecutive_stack_overflows,
  548:         )
  549:         _log_trigger(reasons, "skipped_stack_overflow", tier=tier)
  550:         return False
  551: 
  552:     # Check if Layer 2 is enabled — allows running data loop without Claude quota
  553:     try:
  554:         config = _load_config()
  555:         l2_cfg = config.get("layer2", {})
  556:         if not l2_cfg.get("enabled", True):
  557:             logger.info("Layer 2 disabled (config.layer2.enabled=false), skipping")
  558:             return False
  559:     except Exception as e:
  560:         logger.warning("Failed to load config for layer2 check: %s", e)
  561: 
  562:     tier_cfg = TIER_CONFIG.get(tier, TIER_CONFIG[3])
  563:     timeout = tier_cfg["timeout"]
  564: 
  565:     # 2026-05-05: this reentrancy block reads/mutates the same _agent_proc /
  566:     # _agent_log / _agent_start state that the watchdog tick observes via
  567:     # _check_agent_completion_locked. Without the lock, the watchdog could
  568:     # observe a freshly-killed _agent_proc.poll() exit code and write a
  569:     # "failed"/"incomplete" row at the same time _kill_overrun_agent is
  570:     # writing its "timeout" row — exactly the double-log the lock was added
  571:     # to prevent. Hold _completion_lock for the entire read-decide-kill
  572:     # path; _kill_overrun_agent itself does NOT take the lock so this is
  573:     # safe (no reentrant acquire).
  574:     with _completion_lock:
  575:         if _agent_proc and _agent_proc.poll() is None:
  576:             # BUG-203: use monotonic clock for elapsed — wall clock is NTP-jump-prone.
  577:             # P2B (2026-04-17): via _safe_elapsed_s() so a poisoned _agent_start
  578:             # can't cause a negative elapsed that silently skips the timeout.
  579:             elapsed = _safe_elapsed_s()
  580:             if elapsed > _agent_timeout:
  581:                 # P1B (2026-04-17): helper so check_agent_completion can share
  582:                 # the kill path — see _kill_overrun_agent docstring.
  583:                 kill_ok = _kill_overrun_agent(
  584:                     fallback_reasons=reasons, fallback_tier=tier,
  585:                 )
  586:                 # BUG-92: If kill failed, don't spawn new agent (old one may
  587:                 # still be running)
  588:                 if not kill_ok:
  589:                     logger.error(
  590:                         "Not spawning new agent — old process may still be running"
  591:                     )
  592:                     return False
  593:             else:
  594:                 logger.info(
  595:                     "Agent still running (pid %s, %.0fs), skipping",
  596:                     _agent_proc.pid, elapsed,
  597:                 )
  598:                 return False
  599: 
  600:     if _agent_log:
  601:         _agent_log.close()
  602:         _agent_log = None
  603: 
  604:     try:
  605:         from portfolio.journal import write_context
  606: 
  607:         n = write_context()
  608:         logger.info("Layer 2 context: %d journal entries", n)
  609:     except Exception as e:
  610:         logger.warning("journal context failed: %s", e)
  780:         except Exception as e:
  781:             logger.warning("Multi-agent failed (%s), falling back to single-agent", e)
  782:             prompt = _build_tier_prompt(tier, reasons)
  783:     else:
  784:         prompt = _build_tier_prompt(tier, reasons)
  785: 
  786:     # BUG-214: Append drawdown context so Layer 2 sees current risk levels.
  787:     if _drawdown_context:
  788:         prompt += "\n\n[RISK DATA]" + _drawdown_context
  789:     # P1-12 (2026-05-02): also surface trade-guard warnings to Layer 2 so
  790:     # it can avoid suggesting actions that the guards would just block in
  791:     # check_overtrading_guards anyway.
  792:     if _guard_context:
  793:         prompt += "\n\n[TRADE GUARDS]" + _guard_context
  794: 
  795:     # Decision feedback loop (2026-05-02 research): inject recent decisions
  796:     # for the trigger ticker so Layer 2 can see its own track record and
  797:     # calibrate (e.g., "I said SELL at $73 — price is now $75, was I wrong?").
  798:     try:
  799:         feedback_ticker = _extract_ticker(reasons)
  800:         _feedback = _build_decision_feedback(feedback_ticker)
  801:         if _feedback:
  802:             prompt += "\n\n" + _feedback
  803:     except Exception as e:
  804:         logger.debug("decision feedback failed (non-fatal): %s", e)
  805: 
  806:     max_turns = tier_cfg["max_turns"]
  807: 
  808:     # Try direct claude invocation first; fall back to bat file for T3
  809:     claude_cmd = shutil.which("claude")
  810:     if claude_cmd:
  811:         # 2026-04-13: DO NOT add `--bare`. It disables OAuth/keychain auth
  812:         # and only accepts ANTHROPIC_API_KEY. This user runs on a Max
  813:         # subscription with no API key, so `--bare` silently breaks every
  814:         # invocation ("Not logged in" to stdout, exit 0). Commit b4bb57d
  815:         # added it on 2026-03-27; removed on 2026-04-13 after 3 weeks of
  816:         # silent Layer 2 failures. See portfolio/claude_gate.py
  817:         # (detect_auth_failure) for the runtime guard.
  818:         cmd = [
  819:             claude_cmd, "-p", prompt,
  820:             "--allowedTools", "Edit,Read,Bash,Write",
  821:             "--max-turns", str(max_turns),
  822:         ]
  823:     else:
  824:         # Fallback: use pf-agent.bat (always Tier 3)
  825:         agent_bat = BASE_DIR / "scripts" / "win" / "pf-agent.bat"
  826:         if not agent_bat.exists():
  827:             logger.warning("Agent script not found at %s", agent_bat)
  828:             return False
  829:         cmd = ["cmd", "/c", str(agent_bat)]
  830:         logger.info("claude not on PATH, falling back to pf-agent.bat (T3)")
  831: 
  832:     log_fh = None
  833:     try:
  834:         agent_log_path = DATA_DIR / "agent.log"
  835:         # Capture the current file size BEFORE opening in append mode, so
  836:         # check_agent_completion() can read only this invocation's output
  837:         # (for auth-error detection) and not the entire log history.
  838:         global _agent_log_start_offset
  839:         _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
  840:         log_fh = open(agent_log_path, "a", encoding="utf-8")
  841:         # Strip Claude Code session markers to avoid "nested session" error
  842:         # when the parent process tree has Claude Code running
  843:         agent_env = os.environ.copy()
  844:         agent_env.pop("CLAUDECODE", None)
  845:         agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
  846:         # Increase Node.js stack size to prevent stack overflow in Claude CLI
  847:         agent_env["NODE_OPTIONS"] = "--stack-size=16384"
  848:         # P2 (2026-04-17): mark this subprocess as headless so CLAUDE.md's
  849:         # STARTUP CHECK protocol doesn't ask "How would you like to proceed?"
  850:         # when it finds unresolved critical_errors.jsonl entries. The agent
  851:         # has no stdin (pipe only), so any prompt that blocks on user input
  852:         # makes it hit the tier timeout with zero work done. The CLAUDE.md
  853:         # conditional turns that into "log the unresolved entries in your
  854:         # journal entry and proceed with the trigger task".
  855:         agent_env["PF_HEADLESS_AGENT"] = "1"
  856:         _agent_proc = subprocess.Popen(
  857:             cmd,
  858:             cwd=str(BASE_DIR),
  859:             stdout=log_fh,
  860:             stderr=subprocess.STDOUT,
 1120: 
 1121: 
 1122: def _check_agent_completion_locked():
 1123:     """Body of ``check_agent_completion``. The caller MUST hold
 1124:     ``_completion_lock``. Split out so the watchdog tick can call into
 1125:     the same code path without re-acquiring the lock recursively.
 1126:     """
 1127:     global _agent_proc, _agent_log, _agent_start, _agent_start_wall
 1128:     global _agent_tier, _agent_reasons, _journal_ts_before, _telegram_ts_before
 1129: 
 1130:     if _agent_proc is None:
 1131:         return None
 1132: 
 1133:     exit_code = _agent_proc.poll()
 1134:     if exit_code is None:
 1135:         # Still running. P1B (2026-04-17): enforce the wall-clock timeout
 1136:         # here too — the lazy check in try_invoke_agent only fires when a
 1137:         # new trigger arrives, so a hung agent could run indefinitely if
 1138:         # no new triggers came through (yesterday: T1 timeout=120s ran
 1139:         # 603s). Share the same kill helper used by try_invoke_agent to
 1140:         # keep kill semantics identical.
 1141:         elapsed = _safe_elapsed_s()
 1142:         if _agent_timeout and elapsed > _agent_timeout:
 1143:             killed_tier = _agent_tier
 1144:             killed_reasons = list(_agent_reasons or [])
 1145:             _kill_overrun_agent()
 1146:             return {
 1147:                 "status": "timeout",
 1148:                 "exit_code": None,
 1149:                 "duration_s": round(elapsed, 1),
 1150:                 "tier": killed_tier,
 1151:                 "reasons": killed_reasons,
 1152:                 "journal_written": False,
 1153:                 "telegram_sent": False,
 1154:                 "completed_at": datetime.now(UTC).isoformat(),
 1155:             }
 1156:         return None
 1157: 
 1158:     # Process has finished — collect completion info.
 1159:     # P2B (2026-04-17): via _safe_elapsed_s() so a poisoned _agent_start
 1160:     # can't produce the negative duration_s seen in yesterday's 13:45:45
 1161:     # auth_failure entry (-1776254571.5, matching time.monotonic() - time.time()).
 1162:     duration_s = round(_safe_elapsed_s(), 1)
 1163:     completed_at = datetime.now(UTC).isoformat()
 1164: 
 1165:     # BUG-97: _last_jsonl_ts can raise OSError if file is locked on Windows
 1300:                 send_or_store(
 1301:                     f"*ALERT* Layer 2 auto-disabled after {_MAX_STACK_OVERFLOWS} "
 1302:                     f"consecutive stack overflows (exit {exit_code}). "
 1303:                     "Claude CLI is crashing — investigate project root.",
 1304:                     config, category="alert",
 1305:                 )
 1306:             except Exception as e:
 1307:                 logger.warning("Stack overflow alert failed: %s", e)
 1308:     else:
 1309:         # BUG-95: Reset counter on any non-stack-overflow completion (success or otherwise).
 1310:         # This prevents false positive auto-disable when the consecutive chain is broken.
 1311:         if _consecutive_stack_overflows > 0:
 1312:             _consecutive_stack_overflows = 0
 1313:             _save_stack_overflow_counter(0)
 1314: 
 1315:     # Clean up
 1316:     if _agent_log:
 1317:         try:
 1318:             _agent_log.close()
 1319:         except Exception as e:
 1320:             logger.warning("Agent log close failed: %s", e)
 1321:     _agent_proc = None
 1322:     _agent_log = None
 1323:     _agent_start = 0
 1324:     _agent_start_wall = 0.0
 1325:     _agent_tier = None
 1326:     _agent_reasons = None
 1327:     _journal_ts_before = None
 1328:     _telegram_ts_before = None
 1329:     _patient_txn_count_before = 0
 1330:     _bold_txn_count_before = 0
 1331: 
 1332:     return result
 1333: 
 1334: 
 1335: def get_completion_stats(hours=24):
 1336:     """Compute rolling completion stats from the invocations log.
 1337: 
 1338:     Args:
 1339:         hours: Number of hours to look back (default 24).
 1340: 
 1341:     Returns:
 1342:         dict with keys: total, success, incomplete, failed, timeout,
 1343:         auth_error, completion_rate.  Returns zeroed stats if no data is
 1344:         available.
 1345: 

codex
I’ve got the first pass of the cited ranges. Next I’m tracing the supporting globals and helper functions to see whether these are real bugs or just speculative readings that ignore the surrounding code.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_agent_timeout|_agent_log_start_offset|_startup_grace_active|_consecutive_stack_overflows|_MAX_STACK_OVERFLOWS|def _kill_overrun_agent|def _scan_agent_log_for_auth_failure|def _safe_elapsed_s|def check_agent_completion|def _load_state|def _save_state|def _today_str|def heartbeat|def load_health|def _market_close_hour_utc|def _eu_market_open_hour_utc\" portfolio\\agent_invocation.py portfolio\\trigger.py portfolio\\health.py portfolio\\market_timing.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\agent_invocation.py'; "'$ranges=@(@{s=1;e=140},@{s=450;e=540},@{s=860;e=930},@{s=1160;e=1335}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\trigger.py'; "'$ranges=@(@{s=1;e=120},@{s=460;e=540}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\market_timing.py'; "'$ranges=@(@{s=1;e=140}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 708ms:
portfolio\health.py:44:def load_health() -> dict:
portfolio\health.py:64:def heartbeat() -> None:
portfolio\trigger.py:69:_startup_grace_active = True  # True until first check_triggers call completes
portfolio\trigger.py:106:def _today_str():
portfolio\trigger.py:110:def _load_state():
portfolio\trigger.py:114:def _save_state(state):
portfolio\trigger.py:161:    global _startup_grace_active
portfolio\trigger.py:170:    if _startup_grace_active and saved_pid != current_pid:
portfolio\trigger.py:197:        _startup_grace_active = False
portfolio\trigger.py:201:    _startup_grace_active = False
portfolio\agent_invocation.py:38:_agent_log_start_offset = 0  # byte offset of agent.log at invoke time, for auth-error scan on completion
portfolio\agent_invocation.py:47:_agent_timeout = 900  # per-invocation timeout (set from tier config)
portfolio\agent_invocation.py:61:_MAX_STACK_OVERFLOWS = 5  # auto-disable after this many consecutive stack overflow crashes
portfolio\agent_invocation.py:171:_consecutive_stack_overflows = _load_stack_overflow_counter()
portfolio\agent_invocation.py:350:def _safe_elapsed_s():
portfolio\agent_invocation.py:361:    disabled the P1B timeout path — `elapsed > _agent_timeout` can never
portfolio\agent_invocation.py:393:def _scan_agent_log_for_auth_failure(label: str, extra_context: dict | None = None) -> bool:
portfolio\agent_invocation.py:428:            f.seek(_agent_log_start_offset)
portfolio\agent_invocation.py:446:def _kill_overrun_agent(fallback_reasons=None, fallback_tier=None):
portfolio\agent_invocation.py:540:    global _agent_proc, _agent_log, _agent_start, _agent_start_wall, _agent_timeout
portfolio\agent_invocation.py:544:    if _consecutive_stack_overflows >= _MAX_STACK_OVERFLOWS:
portfolio\agent_invocation.py:547:            _consecutive_stack_overflows,
portfolio\agent_invocation.py:580:            if elapsed > _agent_timeout:
portfolio\agent_invocation.py:838:        global _agent_log_start_offset
portfolio\agent_invocation.py:839:        _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
portfolio\agent_invocation.py:867:        _agent_timeout = timeout
portfolio\agent_invocation.py:1093:def check_agent_completion():
portfolio\agent_invocation.py:1142:        if _agent_timeout and elapsed > _agent_timeout:
portfolio\agent_invocation.py:1199:    # motivated this detection. We captured _agent_log_start_offset before
portfolio\agent_invocation.py:1284:    global _consecutive_stack_overflows
portfolio\agent_invocation.py:1286:        _consecutive_stack_overflows += 1
portfolio\agent_invocation.py:1287:        _save_stack_overflow_counter(_consecutive_stack_overflows)
portfolio\agent_invocation.py:1291:            exit_code, _consecutive_stack_overflows,
portfolio\agent_invocation.py:1293:        if _consecutive_stack_overflows == _MAX_STACK_OVERFLOWS:
portfolio\agent_invocation.py:1296:                _MAX_STACK_OVERFLOWS,
portfolio\agent_invocation.py:1301:                    f"*ALERT* Layer 2 auto-disabled after {_MAX_STACK_OVERFLOWS} "
portfolio\agent_invocation.py:1311:        if _consecutive_stack_overflows > 0:
portfolio\agent_invocation.py:1312:            _consecutive_stack_overflows = 0
portfolio\market_timing.py:53:def _eu_market_open_hour_utc(dt):
portfolio\market_timing.py:92:def _market_close_hour_utc(dt):

 succeeded in 803ms:
    1: """Smart trigger system — detects meaningful market changes to reduce noise.
    2: 
    3: Layer 1 runs on a 10-minute cadence during every market state (see
    4: ``portfolio/market_timing.py:INTERVAL_MARKET_OPEN``). Layer 2 is invoked when:
    5: - Signal consensus: any ticker NEWLY reaches BUY or SELL from HOLD
    6: - Signal flip sustained for SUSTAINED_CHECKS consecutive cycles (see below)
    7: - Price moved >2% since last trigger
    8: - Fear & Greed crossed extreme threshold (20 or 80)
    9: - Sentiment reversal: sustained for SUSTAINED_CHECKS cycles (filters oscillation)
   10: - Post-trade reassessment: after a BUY/SELL trade
   11: 
   12: No periodic cooldown — Layer 2 is only invoked when Layer 1 detects a
   13: meaningful change. The Tier 3 periodic full review (every 2h market / 4h
   14: off-hours) provides the "heartbeat" via classify_tier(), but only when
   15: another trigger has already fired.
   16: """
   17: 
   18: import logging
   19: import os
   20: import re
   21: import time
   22: from datetime import UTC, datetime
   23: from pathlib import Path
   24: 
   25: from portfolio.file_utils import atomic_write_json, load_json
   26: 
   27: logger = logging.getLogger("portfolio.trigger")
   28: 
   29: BASE_DIR = Path(__file__).resolve().parent.parent
   30: STATE_FILE = BASE_DIR / "data" / "trigger_state.json"
   31: PORTFOLIO_FILE = BASE_DIR / "data" / "portfolio_state.json"
   32: PORTFOLIO_BOLD_FILE = BASE_DIR / "data" / "portfolio_state_bold.json"
   33: 
   34: PRICE_THRESHOLD = 0.02  # 2% move
   35: FG_THRESHOLDS = (20, 80)  # extreme fear / extreme greed boundaries
   36: # A signal flip triggers Layer 2 when EITHER of these holds:
   37: #   - SUSTAINED_CHECKS consecutive cycles show the new action, OR
   38: #   - SUSTAINED_DURATION_S seconds of wall-clock time have elapsed since
   39: #     the flip first appeared.
   40: # The count path is the original behavior (unchanged at the 60s cadence).
   41: # The duration path is new (added 2026-04-09 with the cadence bump to 600s);
   42: # at 600s cadence the count path would require ≥30 min of sustained flip
   43: # before triggering, which effectively disables the trigger for fast-moving
   44: # events. The duration gate bounds the worst case to ~1 cycle after flip
   45: # (≈10 min at 600s cadence, ≈2 min at 60s cadence — both unchanged or better
   46: # than the old count-only behavior).
   47: SUSTAINED_CHECKS = 3
   48: SUSTAINED_DURATION_S = 120
   49: 
   50: # Per-ticker flip cooldown (2026-05-08): after a sustained flip fires a Layer 2
   51: # trigger, suppress further sustained flip triggers for the SAME ticker for
   52: # FLIP_COOLDOWN_S seconds.  Prevents whiplash where volatile tickers (e.g. MSTR)
   53: # produce 3+ sustained flips in under an hour, each invoking Layer 2 for a HOLD.
   54: # Does NOT suppress consensus triggers (section 1), price moves (section 3), or
   55: # F&G crossings (section 4) — only section-2 sustained flips.
   56: FLIP_COOLDOWN_S = 1800  # 30 min
   57: 
   58: # Ranging regime dampening (2026-04-22): when a ticker's regime is "ranging",
   59: # require a minimum consensus confidence before triggering Layer 2. In ranging
   60: # markets, consensus oscillates between HOLD and weak BUY/SELL, producing 20+
   61: # Layer 2 invocations per day that all return HOLD — wasting compute and token
   62: # budget. Setting this to 0.0 disables dampening without code change.
   63: RANGING_CONSENSUS_MIN_CONFIDENCE = 0.35
   64: 
   65: # Startup grace period — after a restart, the first loop iteration updates the
   66: # baseline without triggering Layer 2. This prevents spurious T3 full reviews
   67: # every time the loop is restarted for a code update.
   68: _GRACE_PERIOD_KEY = "last_loop_pid"  # stored in trigger_state.json
   69: _startup_grace_active = True  # True until first check_triggers call completes
   70: 
   71: 
   72: def _update_sustained(
   73:     state_dict: dict, key: str, value, now_ts: float
   74: ) -> tuple[bool, bool]:
   75:     """Update sustained-debounce state for a key and return gate results.
   76: 
   77:     Shared by signal flip (section 2) and sentiment reversal (section 5).
   78:     Increments count if value unchanged, resets if changed. Returns
   79:     (count_ok, duration_ok) indicating whether either debounce gate passed.
   80: 
   81:     Duration tracking uses time.monotonic() internally to avoid NTP-jump
   82:     false negatives. On process restart, monotonic origin resets and the
   83:     duration gate conservatively starts fresh (correct behavior — a
   84:     restart already resets the sustained counter).
   85:     """
   86:     mono_now = time.monotonic()
   87:     prev = state_dict.get(key, {})
   88:     if prev.get("value") == value:
   89:         state_dict[key] = {
   90:             "value": value,
   91:             "count": prev["count"] + 1,
   92:             "_mono_start": prev.get("_mono_start", mono_now),
   93:         }
   94:     else:
   95:         state_dict[key] = {
   96:             "value": value,
   97:             "count": 1,
   98:             "_mono_start": mono_now,
   99:         }
  100:     entry = state_dict[key]
  101:     count_ok = entry["count"] >= SUSTAINED_CHECKS
  102:     duration_ok = (mono_now - entry["_mono_start"]) >= SUSTAINED_DURATION_S
  103:     return count_ok, duration_ok
  104: 
  105: 
  106: def _today_str():
  107:     return datetime.now(UTC).strftime("%Y-%m-%d")
  108: 
  109: 
  110: def _load_state():
  111:     return load_json(STATE_FILE, default={})
  112: 
  113: 
  114: def _save_state(state):
  115:     # Prune triggered_consensus entries for tickers not in current signals
  116:     # to prevent unbounded growth when tickers are removed from tracking
  117:     tc = state.get("triggered_consensus", {})
  118:     current_tickers = state.get("_current_tickers")
  119:     if current_tickers is not None:
  120:         removed = {k for k in tc if k not in current_tickers}
  460: 
  461:     if any("F&G crossed" in r for r in reasons):
  462:         return 3
  463: 
  464:     if market_open and hours_since >= _FULL_REVIEW_MARKET_HOURS:
  465:         return 3
  466:     if not market_open and hours_since >= _FULL_REVIEW_OFF_HOURS:
  467:         return 1  # T1 quick check only — save T3 budget for market hours
  468: 
  469:     # Tier 2: new actionable signals
  470:     tier2_patterns = ["consensus", "moved", "post-trade", "flipped"]
  471:     if any(p in r for r in reasons for p in tier2_patterns):
  472:         # Option P (2026-04-17): downshift T2 -> T1 when every reason is
  473:         # low-conviction (consensus <40% confidence or *->HOLD fade flip).
  474:         # Preserves trigger firing + signal/accuracy data; only cuts Claude
  475:         # analysis depth on signals that reliably return HOLD anyway.
  476:         if _should_downshift_to_t1(reasons):
  477:             return 1
  478:         return 2
  479: 
  480:     # Tier 1: cooldowns, sentiment noise, repeated triggers
  481:     return 1
  482: 
  483: 
  484: def update_tier_state(tier, state=None):
  485:     """Update trigger state after a tier classification.
  486: 
  487:     Called by the main loop after classify_tier() to persist tier-specific state.
  488:     M10/NEW-4: accepts an optional state dict to avoid re-reading trigger_state.json.
  489:     """
  490:     if state is None:
  491:         state = _load_state()
  492:     if tier == 3:
  493:         state["last_full_review_time"] = time.time()
  494:     _save_state(state)

 succeeded in 814ms:
    1: """Market timing utilities — DST-aware NYSE and EU hours, market state detection.
    2: 
    3: Includes US (NYSE) and Swedish (Nasdaq Stockholm / Avanza) holiday calendars
    4: so the system skips stock/warrant processing on public holidays, not just weekends.
    5: """
    6: 
    7: from datetime import UTC, date, datetime, timedelta
    8: 
    9: from portfolio.tickers import CRYPTO_SYMBOLS, METALS_SYMBOLS, STOCK_SYMBOLS, SYMBOLS
   10: 
   11: # Backward compat: MARKET_OPEN_HOUR kept at 7 (summer value).
   12: # Callers that need DST-aware EU open should use _eu_market_open_hour_utc().
   13: MARKET_OPEN_HOUR = 7
   14: 
   15: # Loop intervals by market state.
   16: # 2026-04-09: all bumped to 600s (10 min). The reduced 5-ticker universe +
   17: # warm fingpt daemon means we no longer need a 60s fast tick — giving the
   18: # loop breathing room eliminates cadence overruns without losing anything
   19: # meaningful. Weekend was already 600s. See docs/PLAN_FINGPT_DAEMON.md.
   20: INTERVAL_MARKET_OPEN = 600    # 10 min — previously 60s (pre-daemon era)
   21: INTERVAL_MARKET_CLOSED = 600  # 10 min — previously 120s
   22: INTERVAL_WEEKEND = 600        # 10 min — unchanged
   23: 
   24: # States where US stock markets are NOT trading — use this tuple instead of
   25: # hardcoding ("closed", "weekend") to avoid missing the "holiday" state.
   26: MARKET_CLOSED_STATES = ("closed", "weekend", "holiday")
   27: 
   28: 
   29: def _is_eu_dst(dt):
   30:     """Check if a UTC datetime falls within EU Summer Time (CEST).
   31: 
   32:     EU DST rule:
   33:       Starts: last Sunday of March at 01:00 UTC
   34:       Ends:   last Sunday of October at 01:00 UTC
   35: 
   36:     Returns True during CEST (summer), False during CET (winter).
   37:     """
   38:     year = dt.year
   39: 
   40:     # Last Sunday of March
   41:     mar31 = date(year, 3, 31)
   42:     last_sun_mar = 31 - (mar31.weekday() + 1) % 7
   43:     eu_dst_start = datetime(year, 3, last_sun_mar, 1, 0, tzinfo=UTC)
   44: 
   45:     # Last Sunday of October
   46:     oct31 = date(year, 10, 31)
   47:     last_sun_oct = 31 - (oct31.weekday() + 1) % 7
   48:     eu_dst_end = datetime(year, 10, last_sun_oct, 1, 0, tzinfo=UTC)
   49: 
   50:     return eu_dst_start <= dt < eu_dst_end
   51: 
   52: 
   53: def _eu_market_open_hour_utc(dt):
   54:     """Return the EU market open hour in UTC, adjusted for EU DST.
   55: 
   56:     H47: London/Frankfurt open at 08:00 local time.
   57:     CEST (summer, BST=UTC+1): 08:00 local = 07:00 UTC
   58:     CET (winter, GMT=UTC+0): 08:00 local = 08:00 UTC
   59: 
   60:     Previously hardcoded to 7 UTC year-round, which missed the winter hour.
   61:     """
   62:     if _is_eu_dst(dt):
   63:         return 7
   64:     return 8
   65: 
   66: 
   67: def _is_us_dst(dt):
   68:     """Check if a UTC datetime falls within US Eastern Daylight Time (EDT).
   69: 
   70:     US DST rule (since 2007):
   71:       Starts: second Sunday of March at 02:00 local (07:00 UTC)
   72:       Ends:   first Sunday of November at 02:00 local (06:00 UTC)
   73: 
   74:     Returns True during EDT (Mar-Nov), False during EST (Nov-Mar).
   75:     """
   76:     year = dt.year
   77: 
   78:     # Second Sunday of March
   79:     mar1_wd = date(year, 3, 1).weekday()  # 0=Mon..6=Sun
   80:     first_sun_mar = 1 + (6 - mar1_wd) % 7
   81:     second_sun_mar = first_sun_mar + 7
   82:     dst_start = datetime(year, 3, second_sun_mar, 7, 0, tzinfo=UTC)
   83: 
   84:     # First Sunday of November
   85:     nov1_wd = date(year, 11, 1).weekday()
   86:     first_sun_nov = 1 + (6 - nov1_wd) % 7
   87:     dst_end = datetime(year, 11, first_sun_nov, 6, 0, tzinfo=UTC)
   88: 
   89:     return dst_start <= dt < dst_end
   90: 
   91: 
   92: def _market_close_hour_utc(dt):
   93:     """Return the NYSE close hour in UTC, adjusted for DST.
   94: 
   95:     NYSE closes at 16:00 ET.
   96:     EDT (Mar-Nov): 16:00 ET = 20:00 UTC
   97:     EST (Nov-Mar): 16:00 ET = 21:00 UTC
   98:     """
   99:     if _is_us_dst(dt):
  100:         return 20
  101:     return 21
  102: 
  103: 
  104: # ---------------------------------------------------------------------------
  105: # Holiday calendars
  106: # ---------------------------------------------------------------------------
  107: 
  108: 
  109: def _easter_sunday(year):
  110:     """Compute Easter Sunday for a given year using the Anonymous Gregorian algorithm."""
  111:     a = year % 19
  112:     b, c = divmod(year, 100)
  113:     d, e = divmod(b, 4)
  114:     f = (b + 8) // 25
  115:     g = (b - f + 1) // 3
  116:     h = (19 * a + b - d - g + 15) % 30
  117:     i, k = divmod(c, 4)
  118:     l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
  119:     m = (a + 11 * h + 22 * l) // 451
  120:     month, day = divmod(h + l - 7 * m + 114, 31)
  121:     return date(year, month, day + 1)
  122: 
  123: 
  124: def _observed(d):
  125:     """Return the NYSE-observed date for a fixed holiday.
  126: 
  127:     If the holiday falls on Saturday, NYSE observes it Friday.
  128:     If Sunday, NYSE observes it Monday.
  129:     """
  130:     if d.weekday() == 5:  # Saturday
  131:         return d - timedelta(days=1)
  132:     if d.weekday() == 6:  # Sunday
  133:         return d + timedelta(days=1)
  134:     return d
  135: 
  136: 
  137: def _nth_weekday(year, month, weekday, n):
  138:     """Return the nth occurrence of a weekday in a given month.
  139: 
  140:     weekday: 0=Mon, 6=Sun.  n: 1-based (1=first, 2=second, etc.)

 succeeded in 859ms:
    1: """Layer 2 agent invocation — manages the Claude Code subprocess lifecycle."""
    2: 
    3: import logging
    4: import os
    5: import platform
    6: import shutil
    7: import subprocess
    8: import threading
    9: import time
   10: from contextlib import suppress
   11: from datetime import UTC, datetime
   12: from pathlib import Path
   13: 
   14: from portfolio.api_utils import load_config as _load_config
   15: from portfolio.claude_gate import detect_auth_failure
   16: from portfolio.file_utils import atomic_append_jsonl, last_jsonl_entry, load_jsonl
   17: from portfolio.message_store import send_or_store
   18: from portfolio.telegram_notifications import escape_markdown_v1
   19: 
   20: logger = logging.getLogger("portfolio.agent")
   21: 
   22: BASE_DIR = Path(__file__).resolve().parent.parent
   23: DATA_DIR = BASE_DIR / "data"
   24: INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
   25: JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
   26: TELEGRAM_FILE = DATA_DIR / "telegram_messages.jsonl"
   27: PATIENT_PORTFOLIO = DATA_DIR / "portfolio_state.json"
   28: BOLD_PORTFOLIO = DATA_DIR / "portfolio_state_bold.json"
   29: 
   30: # BUG-214: Drawdown circuit breaker thresholds.
   31: # Advisory at WARN level, hard-block at BLOCK level.
   32: # User accepts 10-20% knockout risk; only de-risk at 50%+.
   33: _DRAWDOWN_WARN_PCT = 20.0
   34: _DRAWDOWN_BLOCK_PCT = 50.0
   35: 
   36: _agent_proc = None
   37: _agent_log = None
   38: _agent_log_start_offset = 0  # byte offset of agent.log at invoke time, for auth-error scan on completion
   39: _agent_start = 0
   40: # P2B follow-up (Codex P2 #2, 2026-04-17): fallback wall-clock timestamp
   41: # for timeout enforcement when `_agent_start` (monotonic) gets poisoned.
   42: # The clamp alone could silently disable the P1B T1 timeout check; this
   43: # fallback lets _safe_elapsed_s() recover a plausible elapsed from wall
   44: # clock so the hung agent still gets killed. Always set alongside
   45: # _agent_start so the pair are in sync.
   46: _agent_start_wall = 0.0
   47: _agent_timeout = 900  # per-invocation timeout (set from tier config)
   48: _agent_tier = None  # tier of the currently running agent
   49: _agent_reasons = None  # trigger reasons for the current invocation
   50: _journal_ts_before = None  # last journal timestamp before agent started
   51: _telegram_ts_before = None  # last telegram timestamp before agent started
   52: 
   53: # BUG-219: Transaction counts at invoke time — used by check_agent_completion()
   54: # to detect new trades and call record_trade() for overtrading prevention.
   55: # PR-R4-4: record_trade() was never called from production code; this wires it.
   56: _patient_txn_count_before = 0
   57: _bold_txn_count_before = 0
   58: 
   59: # Stack overflow detection — exit code 3221225794 = Windows STATUS_STACK_OVERFLOW (0xC00000FD)
   60: _STACK_OVERFLOW_EXIT_CODE = 3221225794
   61: _MAX_STACK_OVERFLOWS = 5  # auto-disable after this many consecutive stack overflow crashes
   62: _STACK_OVERFLOW_FILE = DATA_DIR / "stack_overflow_counter.json"
   63: 
   64: # 2026-05-05 (item 3a of dashboard-noise-followups, see
   65: # docs/plans/2026-05-05-l2-completion-watchdog.md): completion-detection
   66: # watchdog. ``check_agent_completion`` is the only path that observes
   67: # subprocess.poll() and enforces the per-tier wall-clock timeout, but it
   68: # was called only once per ``main.run()`` cycle. When the cycle bloats
   69: # (333-918s violations 2026-05-01..04), a T1 subprocess that finishes
   70: # at its real 120s budget is not noticed for up to 6 more minutes —
   71: # inflating ``duration_s`` in invocations.jsonl and delaying the kill
   72: # of a hung agent past its real budget. The daemon thread below runs
   73: # the same check every 30s independent of ``run()``'s cadence; the
   74: # lock serialises with the main-thread call site so the two cannot
   75: # race on ``_agent_proc`` / ``_agent_start`` state.
   76: _COMPLETION_WATCHDOG_INTERVAL_S = 30
   77: _completion_lock = threading.Lock()
   78: _watchdog_thread: threading.Thread | None = None
   79: _watchdog_stop = threading.Event()
   80: 
   81: 
   82: def _completion_watchdog() -> None:
   83:     """Daemon thread body: poll completion every 30 s while not stopped.
   84: 
   85:     Each tick takes ``_completion_lock`` and calls
   86:     ``_check_agent_completion_locked`` directly so the main-thread call
   87:     via ``check_agent_completion`` and this watchdog tick share one
   88:     critical section. Failures inside the tick are logged and swallowed
   89:     — the watchdog must never die from a transient I/O error or it
   90:     silently regresses to the pre-fix state where the main loop is the
   91:     only completion observer.
   92:     """
   93:     while not _watchdog_stop.is_set():
   94:         # Event.wait(timeout) returns True if the event was set during
   95:         # the wait — ie shutdown. Returning False means the timeout
   96:         # elapsed normally, so we tick.
   97:         if _watchdog_stop.wait(_COMPLETION_WATCHDOG_INTERVAL_S):
   98:             return
   99:         try:
  100:             with _completion_lock:
  101:                 _check_agent_completion_locked()
  102:         except Exception as e:  # noqa: BLE001 — never let the watchdog die
  103:             logger.warning("completion watchdog tick failed: %s", e)
  104: 
  105: 
  106: def _ensure_completion_watchdog() -> None:
  107:     """Start the daemon watchdog if it is not already running.
  108: 
  109:     Idempotent: the spawn happens at most once per process under normal
  110:     operation. If the previous thread died (uncaught exception escaping
  111:     the ``except Exception`` above is impossible, but a thread.start
  112:     failure or interpreter restart between calls could leave the global
  113:     pointing at a dead thread), spawn a fresh one. Resets the stop
  114:     event so a successor process that imports this module after a
  115:     SIGTERM can still arm a new watchdog.
  116: 
  117:     Uses ``_completion_lock`` to make the is-alive-check + spawn atomic
  118:     so concurrent callers cannot both pass the check and both spawn (a
  119:     race exposed by tests that drive start/stop concurrently — in
  120:     production the lazy-start happens once at the end of try_invoke_agent,
  121:     which is itself serialised by the main loop).
  122:     """
  123:     global _watchdog_thread
  124:     with _completion_lock:
  125:         if _watchdog_thread is not None and _watchdog_thread.is_alive():
  126:             return
  127:         _watchdog_stop.clear()
  128:         _watchdog_thread = threading.Thread(
  129:             target=_completion_watchdog,
  130:             name="L2CompletionWatchdog",
  131:             daemon=True,
  132:         )
  133:         _watchdog_thread.start()
  134: 
  135: 
  136: def _stop_completion_watchdog(timeout_s: float = 1.0) -> None:
  137:     """Signal the watchdog to exit and wait briefly for it.
  138: 
  139:     Used by tests to keep xdist parallel runs hermetic; production code
  140:     relies on ``daemon=True`` to terminate the thread at interpreter
  450:     be called from ``check_agent_completion``. Previously the timeout
  451:     check lived only inside try_invoke_agent, meaning a hung agent could
  452:     run indefinitely if no new triggers fired (yesterday evidence: T1
  453:     invoked 16:04:58 with timeout=120s completed at 16:15:01 = 603s).
  454: 
  455:     Logs the trigger with status="timeout" and clears ``_agent_proc`` /
  456:     ``_agent_log`` on the way out.
  457: 
  458:     P1-3 (2026-05-02 last-followups): also scans the captured agent.log
  459:     slice for claude-CLI auth-error markers BEFORE clearing module state,
  460:     so the silent-auth-failure detector covers the timeout path too — not
  461:     just the happy completion path. See ``_scan_agent_log_for_auth_failure``
  462:     for full rationale.
  463: 
  464:     Args:
  465:         fallback_reasons: Reason list to use for the trigger log entry if
  466:             ``_agent_reasons`` is empty (caller context for the missing
  467:             _reasons.).
  468:         fallback_tier: Tier to log if ``_agent_tier`` is None.
  469: 
  470:     Returns:
  471:         bool: True if the kill succeeded (or the process had already
  472:         exited). False if the kill command itself failed — caller must
  473:         NOT spawn a replacement in that case because the old process
  474:         may still be holding resources.
  475:     """
  476:     global _agent_proc, _agent_log
  477: 
  478:     if _agent_proc is None:
  479:         return True
  480: 
  481:     pid = _agent_proc.pid
  482:     elapsed = _safe_elapsed_s()
  483:     logger.info("Agent pid=%s timed out (%.0fs), killing", pid, elapsed)
  484: 
  485:     kill_ok = True
  486:     if platform.system() == "Windows":
  487:         # BUG-92: Check taskkill return code to detect kill failure
  488:         # BUG-189: rc=128 means process already exited — treat as success
  489:         result = subprocess.run(
  490:             ["taskkill", "/F", "/T", "/PID", str(pid)],
  491:             capture_output=True,
  492:         )
  493:         if result.returncode not in (0, 128):
  494:             logger.error(
  495:                 "taskkill failed (rc=%d): %s",
  496:                 result.returncode,
  497:                 result.stderr.decode(errors="replace").strip(),
  498:             )
  499:             kill_ok = False
  500:         elif result.returncode == 128:
  501:             logger.info("Agent pid=%s already exited (rc=128)", pid)
  502:     else:
  503:         _agent_proc.kill()
  504:     try:
  505:         _agent_proc.wait(timeout=15)  # BUG-189: 15s for Claude CLI Node.js teardown
  506:     except subprocess.TimeoutExpired:
  507:         if kill_ok:
  508:             logger.error("Agent pid=%s did not exit after kill+15s wait", pid)
  509:         kill_ok = False
  510: 
  511:     if _agent_log:
  512:         try:
  513:             _agent_log.close()
  514:         except Exception as e:
  515:             logger.warning("Agent log close failed: %s", e)
  516:         _agent_log = None
  517: 
  518:     # P1-3 (2026-05-02 last-followups): scan the captured agent.log slice
  519:     # for auth-error markers before forgetting the dead subprocess. Done
  520:     # AFTER closing _agent_log (so any buffered output is flushed) but
  521:     # BEFORE _agent_proc / _agent_tier / _agent_reasons are cleared (so
  522:     # the auth-failure record carries the right tier + trigger context).
  523:     # Best-effort: failures are swallowed inside the helper so a busted
  524:     # log read can never break the kill path.
  525:     auth_label = f"layer2_t{_agent_tier}_timeout" if _agent_tier else "layer2_timeout"
  526:     _scan_agent_log_for_auth_failure(auth_label)
  527: 
  528:     # BUG-91: Log the timed-out invocation before returning
  529:     _log_trigger(
  530:         _agent_reasons or fallback_reasons or [],
  531:         "timeout",
  532:         tier=_agent_tier or fallback_tier,
  533:     )
  534: 
  535:     _agent_proc = None
  536:     return kill_ok
  537: 
  538: 
  539: def invoke_agent(reasons, tier=3):
  540:     global _agent_proc, _agent_log, _agent_start, _agent_start_wall, _agent_timeout
  860:             stderr=subprocess.STDOUT,
  861:             env=agent_env,
  862:         )
  863:         _agent_log = log_fh  # transfer ownership on success
  864:         log_fh = None  # prevent cleanup below from closing it
  865:         _agent_start = time.monotonic()
  866:         _agent_start_wall = time.time()  # wall-clock fallback for P2B
  867:         _agent_timeout = timeout
  868:         _agent_tier = tier
  869:         _agent_reasons = list(reasons)
  870:         _journal_ts_before = _safe_last_jsonl_ts(JOURNAL_FILE, "journal")
  871:         _telegram_ts_before = _safe_last_jsonl_ts(TELEGRAM_FILE, "telegram")
  872:         # BUG-219: Snapshot transaction counts so check_agent_completion()
  873:         # can detect new trades and call record_trade().
  874:         global _patient_txn_count_before, _bold_txn_count_before
  875:         try:
  876:             from portfolio.file_utils import load_json
  877:             _patient_txn_count_before = len(
  878:                 (load_json(PATIENT_PORTFOLIO, default={}) or {}).get("transactions", [])
  879:             )
  880:             _bold_txn_count_before = len(
  881:                 (load_json(BOLD_PORTFOLIO, default={}) or {}).get("transactions", [])
  882:             )
  883:         except Exception:
  884:             _patient_txn_count_before = 0
  885:             _bold_txn_count_before = 0
  886:         # 2026-04-17: Publish the tier into health_state so loop_contract
  887:         # can pick the right per-tier grace window for the journal-activity
  888:         # check. Without this, the contract defaults to T3 grace (20m),
  889:         # which is conservative but can delay detection when an all-T1
  890:         # cadence runs silent. See loop_contract._get_layer2_grace_s() for
  891:         # the consumer and LAYER2_JOURNAL_GRACE_S_BY_TIER for the table.
  892:         # Best-effort: never fail the invocation because health_state is
  893:         # unwriteable (atomic_write_json handles the happy path; any
  894:         # exception is logged and swallowed).
  895:         try:
  896:             from portfolio.file_utils import atomic_write_json, load_json
  897:             # 2026-04-17 Codex P2: when claude is missing from PATH we fall
  898:             # back to pf-agent.bat which is unconditionally T3 regardless of
  899:             # the requested tier. Record the *effective* tier so the
  900:             # per-tier grace window in loop_contract reflects what's
  901:             # actually running.
  902:             effective_tier = 3 if not claude_cmd else tier
  903:             health_path = DATA_DIR / "health_state.json"
  904:             health = load_json(health_path, default={}) or {}
  905:             health["last_invocation_tier"] = effective_tier
  906:             health["last_invocation_tier_ts"] = datetime.now(UTC).isoformat()
  907:             atomic_write_json(health_path, health)
  908:         except Exception as e:
  909:             logger.warning("health_state tier publish failed: %s", e)
  910:         logger.info(
  911:             "Agent T%d invoked pid=%s max_turns=%d timeout=%ds (%s)",
  912:             tier, _agent_proc.pid, max_turns, timeout,
  913:             ", ".join(reasons[:3]),
  914:         )
  915:         # 2026-05-05: arm the completion watchdog so the wall-clock
  916:         # timeout fires within ~30 s of the real budget even when the
  917:         # main loop's run() cycle bloats. See module-level note at
  918:         # _COMPLETION_WATCHDOG_INTERVAL_S.
  919:         _ensure_completion_watchdog()
  920:         # Save Layer 2 invocation notification (save-only, not sent to Telegram)
  921:         try:
  922:             config = _load_config()
  923:             reason_str = escape_markdown_v1(", ".join(reasons[:3]))
  924:             if len(reasons) > 3:
  925:                 reason_str += f" (+{len(reasons) - 3} more)"
  926:             tier_label = tier_cfg["label"]
  927:             notify_msg = f"_Layer 2 T{tier} ({tier_label}): {reason_str}_"
  928:             send_or_store(notify_msg, config, category="invocation")
  929:         except Exception as e:
  930:             logger.warning("invocation notification failed: %s", e)
 1160:     # can't produce the negative duration_s seen in yesterday's 13:45:45
 1161:     # auth_failure entry (-1776254571.5, matching time.monotonic() - time.time()).
 1162:     duration_s = round(_safe_elapsed_s(), 1)
 1163:     completed_at = datetime.now(UTC).isoformat()
 1164: 
 1165:     # BUG-97: _last_jsonl_ts can raise OSError if file is locked on Windows
 1166:     try:
 1167:         journal_ts_after = _last_jsonl_ts(JOURNAL_FILE)
 1168:     except Exception:
 1169:         logger.warning("Failed to read journal timestamp after agent completion")
 1170:         journal_ts_after = None
 1171:     journal_written = (
 1172:         _journal_ts_before is not None
 1173:         and journal_ts_after is not None
 1174:         and journal_ts_after != _journal_ts_before
 1175:     )
 1176: 
 1177:     # BUG-97: Same protection for telegram file
 1178:     try:
 1179:         telegram_ts_after = _last_jsonl_ts(TELEGRAM_FILE)
 1180:     except Exception:
 1181:         logger.warning("Failed to read telegram timestamp after agent completion")
 1182:         telegram_ts_after = None
 1183:     telegram_sent = (
 1184:         _telegram_ts_before is not None
 1185:         and telegram_ts_after is not None
 1186:         and telegram_ts_after != _telegram_ts_before
 1187:     )
 1188: 
 1189:     # Without a baseline from invoke_agent(), stay conservative and do not infer
 1190:     # success from pre-existing files in the workspace.
 1191:     if _journal_ts_before is None:
 1192:         journal_written = False
 1193:     if _telegram_ts_before is None:
 1194:         telegram_sent = False
 1195: 
 1196:     # 2026-04-13: Scan agent.log for auth-error markers (see claude_gate.py
 1197:     # detect_auth_failure). Claude CLI can exit 0 while printing "Not logged
 1198:     # in" to stdout — that's exactly the 3-week silent Layer 2 outage that
 1199:     # motivated this detection. We captured _agent_log_start_offset before
 1200:     # spawning the subprocess, so we only scan output from this invocation.
 1201:     #
 1202:     # P1-3 (2026-05-02 last-followups): scan logic extracted to
 1203:     # ``_scan_agent_log_for_auth_failure`` so the timeout-kill path
 1204:     # (``_kill_overrun_agent``) can share the exact same semantics. Without
 1205:     # the helper, fixing one path and forgetting the other would re-open
 1206:     # the same asymmetry the timeout path used to have.
 1207:     auth_error_detected = _scan_agent_log_for_auth_failure(
 1208:         f"layer2_t{_agent_tier}",
 1209:         extra_context={"exit_code": exit_code, "duration_s": duration_s},
 1210:     )
 1211: 
 1212:     # Determine status
 1213:     if auth_error_detected:
 1214:         status = "auth_error"
 1215:     elif exit_code != 0:
 1216:         status = "failed"
 1217:     elif journal_written and telegram_sent:
 1218:         status = "success"
 1219:     else:
 1220:         status = "incomplete"
 1221: 
 1222:     result = {
 1223:         "status": status,
 1224:         "exit_code": exit_code,
 1225:         "duration_s": duration_s,
 1226:         "tier": _agent_tier,
 1227:         # Codex P2 #3 follow-up (2026-04-17): include `reasons` so the
 1228:         # completion-path and timeout-path dicts have symmetric shape.
 1229:         # Callers that dispatch on reasons shouldn't need to know which
 1230:         # path produced the dict.
 1231:         "reasons": list(_agent_reasons or []),
 1232:         "completed_at": completed_at,
 1233:         "journal_written": journal_written,
 1234:         "telegram_sent": telegram_sent,
 1235:     }
 1236: 
 1237:     # Log to invocations file
 1238:     log_entry = {
 1239:         "ts": completed_at,
 1240:         "reasons": _agent_reasons or [],
 1241:         "status": status,
 1242:         "tier": _agent_tier,
 1243:         "exit_code": exit_code,
 1244:         "duration_s": duration_s,
 1245:         "journal_written": journal_written,
 1246:         "telegram_sent": telegram_sent,
 1247:     }
 1248:     try:
 1249:         atomic_append_jsonl(INVOCATIONS_FILE, log_entry)
 1250:     except Exception as e:
 1251:         logger.warning("Failed to log agent completion: %s", e)
 1252: 
 1253:     # Post-process: extract fishing context from journal for metals fish engine
 1254:     if journal_written:
 1255:         with suppress(Exception):
 1256:             new_journal_entry = last_jsonl_entry(JOURNAL_FILE)
 1257:             if new_journal_entry:
 1258:                 _write_fishing_context(new_journal_entry)
 1259: 
 1260:     # BUG-219 / PR-R4-4: Wire record_trade() into production.
 1261:     # After a successful agent run, check if new transactions appeared in
 1262:     # either portfolio and record them for overtrading prevention guards
 1263:     # (cooldowns, loss escalation, position rate limits).
 1264:     _record_new_trades()
 1265: 
 1266:     logger.info(
 1267:         "Agent completed: status=%s exit=%d duration=%.1fs tier=%s journal=%s telegram=%s",
 1268:         status, exit_code, duration_s, _agent_tier, journal_written, telegram_sent,
 1269:     )
 1270: 
 1271:     # Telegram alert on any agent failure (not just stack overflow)
 1272:     if status == "failed":
 1273:         try:
 1274:             config = _load_config()
 1275:             send_or_store(
 1276:                 f"*L2 FAILED* T{_agent_tier} exit={exit_code} "
 1277:                 f"({duration_s:.0f}s) journal={journal_written} tg={telegram_sent}",
 1278:                 config, category="error",
 1279:             )
 1280:         except Exception as e:
 1281:             logger.warning("Agent failure alert failed: %s", e)
 1282: 
 1283:     # Track consecutive stack overflow crashes
 1284:     global _consecutive_stack_overflows
 1285:     if exit_code == _STACK_OVERFLOW_EXIT_CODE:
 1286:         _consecutive_stack_overflows += 1
 1287:         _save_stack_overflow_counter(_consecutive_stack_overflows)
 1288:         logger.error(
 1289:             "Claude CLI stack overflow (exit %d), %d consecutive. "
 1290:             "Check project root for problematic files or update Claude Code.",
 1291:             exit_code, _consecutive_stack_overflows,
 1292:         )
 1293:         if _consecutive_stack_overflows == _MAX_STACK_OVERFLOWS:
 1294:             logger.error(
 1295:                 "Layer 2 auto-disabled after %d consecutive stack overflows",
 1296:                 _MAX_STACK_OVERFLOWS,
 1297:             )
 1298:             try:
 1299:                 config = _load_config()
 1300:                 send_or_store(
 1301:                     f"*ALERT* Layer 2 auto-disabled after {_MAX_STACK_OVERFLOWS} "
 1302:                     f"consecutive stack overflows (exit {exit_code}). "
 1303:                     "Claude CLI is crashing — investigate project root.",
 1304:                     config, category="alert",
 1305:                 )
 1306:             except Exception as e:
 1307:                 logger.warning("Stack overflow alert failed: %s", e)
 1308:     else:
 1309:         # BUG-95: Reset counter on any non-stack-overflow completion (success or otherwise).
 1310:         # This prevents false positive auto-disable when the consecutive chain is broken.
 1311:         if _consecutive_stack_overflows > 0:
 1312:             _consecutive_stack_overflows = 0
 1313:             _save_stack_overflow_counter(0)
 1314: 
 1315:     # Clean up
 1316:     if _agent_log:
 1317:         try:
 1318:             _agent_log.close()
 1319:         except Exception as e:
 1320:             logger.warning("Agent log close failed: %s", e)
 1321:     _agent_proc = None
 1322:     _agent_log = None
 1323:     _agent_start = 0
 1324:     _agent_start_wall = 0.0
 1325:     _agent_tier = None
 1326:     _agent_reasons = None
 1327:     _journal_ts_before = None
 1328:     _telegram_ts_before = None
 1329:     _patient_txn_count_before = 0
 1330:     _bold_txn_count_before = 0
 1331: 
 1332:     return result
 1333: 
 1334: 
 1335: def get_completion_stats(hours=24):

codex
The supporting code already weakens a few of Claude’s claims. I’m checking tests and adjacent callers now, because some of these findings depend on lifecycle assumptions that may already be handled elsewhere.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"_agent_log_start_offset|startup grace|triggered_consensus|classify_tier\\(|heartbeat_keepalive|ThreadPoolExecutor|cancel_futures|stack overflow|_build_decision_feedback|_agent_timeout and elapsed|kill_ok|ImportError|claude_cmd\" tests portfolio" in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\health.py'; "'$ranges=@(@{s=1;e=90}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\main.py'; "'$ranges=@(@{s=580;e=665}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-10T15:26:23.990893Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
portfolio\agent_invocation.py:38:_agent_log_start_offset = 0  # byte offset of agent.log at invoke time, for auth-error scan on completion
portfolio\agent_invocation.py:61:_MAX_STACK_OVERFLOWS = 5  # auto-disable after this many consecutive stack overflow crashes
portfolio\agent_invocation.py:154:    """Load persisted stack overflow counter. Returns 0 if missing/corrupt."""
portfolio\agent_invocation.py:163:    """Persist stack overflow counter to survive loop restarts."""
portfolio\agent_invocation.py:282:def _build_decision_feedback(ticker, max_entries=5):
portfolio\agent_invocation.py:428:            f.seek(_agent_log_start_offset)
portfolio\agent_invocation.py:485:    kill_ok = True
portfolio\agent_invocation.py:499:            kill_ok = False
portfolio\agent_invocation.py:507:        if kill_ok:
portfolio\agent_invocation.py:509:        kill_ok = False
portfolio\agent_invocation.py:536:    return kill_ok
portfolio\agent_invocation.py:543:    # Check if Layer 2 is auto-disabled due to consecutive stack overflows
portfolio\agent_invocation.py:546:            "Layer 2 skipped: auto-disabled after %d consecutive stack overflows",
portfolio\agent_invocation.py:583:                kill_ok = _kill_overrun_agent(
portfolio\agent_invocation.py:588:                if not kill_ok:
portfolio\agent_invocation.py:630:    # (ImportError, IO error on portfolio_state.json mid-rename, KeyError
portfolio\agent_invocation.py:637:    # - A complete failure to even load the check (ImportError) is logged
portfolio\agent_invocation.py:800:        _feedback = _build_decision_feedback(feedback_ticker)
portfolio\agent_invocation.py:809:    claude_cmd = shutil.which("claude")
portfolio\agent_invocation.py:810:    if claude_cmd:
portfolio\agent_invocation.py:819:            claude_cmd, "-p", prompt,
portfolio\agent_invocation.py:838:        global _agent_log_start_offset
portfolio\agent_invocation.py:839:        _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
portfolio\agent_invocation.py:846:        # Increase Node.js stack size to prevent stack overflow in Claude CLI
portfolio\agent_invocation.py:902:            effective_tier = 3 if not claude_cmd else tier
portfolio\agent_invocation.py:1142:        if _agent_timeout and elapsed > _agent_timeout:
portfolio\agent_invocation.py:1199:    # motivated this detection. We captured _agent_log_start_offset before
portfolio\agent_invocation.py:1271:    # Telegram alert on any agent failure (not just stack overflow)
portfolio\agent_invocation.py:1283:    # Track consecutive stack overflow crashes
portfolio\agent_invocation.py:1289:            "Claude CLI stack overflow (exit %d), %d consecutive. "
portfolio\agent_invocation.py:1295:                "Layer 2 auto-disabled after %d consecutive stack overflows",
portfolio\agent_invocation.py:1302:                    f"consecutive stack overflows (exit {exit_code}). "
portfolio\health.py:96:class heartbeat_keepalive:  # noqa: N801 — context-manager naming convention
portfolio\health.py:106:        with heartbeat_keepalive():
portfolio\health.py:123:    def __enter__(self) -> "heartbeat_keepalive":
portfolio\health.py:129:            logger.warning("heartbeat_keepalive initial beat failed", exc_info=True)
portfolio\health.py:149:                logger.warning("heartbeat_keepalive tick failed", exc_info=True)
portfolio\main.py:44:except ImportError:
portfolio\main.py:49:except ImportError:
portfolio\main.py:475:    from concurrent.futures import ThreadPoolExecutor, as_completed
portfolio\main.py:610:    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
portfolio\main.py:661:        pool.shutdown(wait=False, cancel_futures=True)
portfolio\main.py:818:        tier = classify_tier(reasons_list)
portfolio\main.py:839:        from portfolio.health import heartbeat_keepalive
portfolio\main.py:847:                with heartbeat_keepalive():
portfolio\main.py:856:            with heartbeat_keepalive():
portfolio\main.py:875:            from portfolio.health import heartbeat_keepalive
portfolio\main.py:876:            with heartbeat_keepalive():
portfolio\main.py:886:        from portfolio.health import heartbeat_keepalive
portfolio\main.py:888:        with heartbeat_keepalive():
portfolio\trigger.py:14:off-hours) provides the "heartbeat" via classify_tier(), but only when
portfolio\trigger.py:115:    # Prune triggered_consensus entries for tickers not in current signals
portfolio\trigger.py:117:    tc = state.get("triggered_consensus", {})
portfolio\trigger.py:124:        state["triggered_consensus"] = pruned
portfolio\trigger.py:191:        # Update triggered_consensus baseline to current state
portfolio\trigger.py:192:        tc = state.get("triggered_consensus", {})
portfolio\trigger.py:195:        state["triggered_consensus"] = tc
portfolio\trigger.py:213:    #    trigger (#2). Uses persistent triggered_consensus that is NOT wiped
portfolio\trigger.py:219:    triggered_consensus = state.get("triggered_consensus", {})
portfolio\trigger.py:222:        last_tc = triggered_consensus.get(ticker, "HOLD")
portfolio\trigger.py:239:                triggered_consensus[ticker] = action
portfolio\trigger.py:243:            triggered_consensus[ticker] = action
portfolio\trigger.py:246:            triggered_consensus[ticker] = "HOLD"
portfolio\trigger.py:250:            triggered_consensus[ticker] = action
portfolio\trigger.py:251:    state["triggered_consensus"] = triggered_consensus
portfolio\trigger.py:348:        # classify_tier() can correctly detect the first real trigger of the day.
portfolio\trigger.py:419:    classify_tier() has already chosen T2 — T1 and T3 are never affected.
portfolio\trigger.py:432:def classify_tier(reasons, state=None):
portfolio\trigger.py:487:    Called by the main loop after classify_tier() to persist tier-specific state.
portfolio\claude_gate.py:8:Direct ``subprocess.Popen([claude_cmd, "-p", ...])`` calls are FORBIDDEN.
portfolio\claude_gate.py:258:def _find_claude_cmd() -> str | None:
portfolio\claude_gate.py:462:    claude_cmd = _find_claude_cmd()
portfolio\claude_gate.py:463:    if not claude_cmd:
portfolio\claude_gate.py:479:        claude_cmd, "-p", prompt,
portfolio\claude_gate.py:579:    claude_cmd = _find_claude_cmd()
portfolio\claude_gate.py:580:    if not claude_cmd:
portfolio\claude_gate.py:585:        claude_cmd, "-p", prompt,
portfolio\multi_agent_layer2.py:136:    claude_cmd = shutil.which("claude")
portfolio\multi_agent_layer2.py:137:    if not claude_cmd:
portfolio\multi_agent_layer2.py:161:            claude_cmd, "-p", prompt,
rg: tests: The system cannot find the file specified. (os error 2)

 exited 1 in 555ms:
portfolio\agent_invocation.py:38:_agent_log_start_offset = 0  # byte offset of agent.log at invoke time, for auth-error scan on completion
portfolio\agent_invocation.py:61:_MAX_STACK_OVERFLOWS = 5  # auto-disable after this many consecutive stack overflow crashes
portfolio\agent_invocation.py:154:    """Load persisted stack overflow counter. Returns 0 if missing/corrupt."""
portfolio\agent_invocation.py:163:    """Persist stack overflow counter to survive loop restarts."""
portfolio\agent_invocation.py:282:def _build_decision_feedback(ticker, max_entries=5):
portfolio\agent_invocation.py:428:            f.seek(_agent_log_start_offset)
portfolio\agent_invocation.py:485:    kill_ok = True
portfolio\agent_invocation.py:499:            kill_ok = False
portfolio\agent_invocation.py:507:        if kill_ok:
portfolio\agent_invocation.py:509:        kill_ok = False
portfolio\agent_invocation.py:536:    return kill_ok
portfolio\agent_invocation.py:543:    # Check if Layer 2 is auto-disabled due to consecutive stack overflows
portfolio\agent_invocation.py:546:            "Layer 2 skipped: auto-disabled after %d consecutive stack overflows",
portfolio\agent_invocation.py:583:                kill_ok = _kill_overrun_agent(
portfolio\agent_invocation.py:588:                if not kill_ok:
portfolio\agent_invocation.py:630:    # (ImportError, IO error on portfolio_state.json mid-rename, KeyError
portfolio\agent_invocation.py:637:    # - A complete failure to even load the check (ImportError) is logged
portfolio\agent_invocation.py:800:        _feedback = _build_decision_feedback(feedback_ticker)
portfolio\agent_invocation.py:809:    claude_cmd = shutil.which("claude")
portfolio\agent_invocation.py:810:    if claude_cmd:
portfolio\agent_invocation.py:819:            claude_cmd, "-p", prompt,
portfolio\agent_invocation.py:838:        global _agent_log_start_offset
portfolio\agent_invocation.py:839:        _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
portfolio\agent_invocation.py:846:        # Increase Node.js stack size to prevent stack overflow in Claude CLI
portfolio\agent_invocation.py:902:            effective_tier = 3 if not claude_cmd else tier
portfolio\agent_invocation.py:1142:        if _agent_timeout and elapsed > _agent_timeout:
portfolio\agent_invocation.py:1199:    # motivated this detection. We captured _agent_log_start_offset before
portfolio\agent_invocation.py:1271:    # Telegram alert on any agent failure (not just stack overflow)
portfolio\agent_invocation.py:1283:    # Track consecutive stack overflow crashes
portfolio\agent_invocation.py:1289:            "Claude CLI stack overflow (exit %d), %d consecutive. "
portfolio\agent_invocation.py:1295:                "Layer 2 auto-disabled after %d consecutive stack overflows",
portfolio\agent_invocation.py:1302:                    f"consecutive stack overflows (exit {exit_code}). "
portfolio\health.py:96:class heartbeat_keepalive:  # noqa: N801 — context-manager naming convention
portfolio\health.py:106:        with heartbeat_keepalive():
portfolio\health.py:123:    def __enter__(self) -> "heartbeat_keepalive":
portfolio\health.py:129:            logger.warning("heartbeat_keepalive initial beat failed", exc_info=True)
portfolio\health.py:149:                logger.warning("heartbeat_keepalive tick failed", exc_info=True)
portfolio\main.py:44:except ImportError:
portfolio\main.py:49:except ImportError:
portfolio\main.py:475:    from concurrent.futures import ThreadPoolExecutor, as_completed
portfolio\main.py:610:    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
portfolio\main.py:661:        pool.shutdown(wait=False, cancel_futures=True)
portfolio\main.py:818:        tier = classify_tier(reasons_list)
portfolio\main.py:839:        from portfolio.health import heartbeat_keepalive
portfolio\main.py:847:                with heartbeat_keepalive():
portfolio\main.py:856:            with heartbeat_keepalive():
portfolio\main.py:875:            from portfolio.health import heartbeat_keepalive
portfolio\main.py:876:            with heartbeat_keepalive():
portfolio\main.py:886:        from portfolio.health import heartbeat_keepalive
portfolio\main.py:888:        with heartbeat_keepalive():
portfolio\trigger.py:14:off-hours) provides the "heartbeat" via classify_tier(), but only when
portfolio\trigger.py:115:    # Prune triggered_consensus entries for tickers not in current signals
portfolio\trigger.py:117:    tc = state.get("triggered_consensus", {})
portfolio\trigger.py:124:        state["triggered_consensus"] = pruned
portfolio\trigger.py:191:        # Update triggered_consensus baseline to current state
portfolio\trigger.py:192:        tc = state.get("triggered_consensus", {})
portfolio\trigger.py:195:        state["triggered_consensus"] = tc
portfolio\trigger.py:213:    #    trigger (#2). Uses persistent triggered_consensus that is NOT wiped
portfolio\trigger.py:219:    triggered_consensus = state.get("triggered_consensus", {})
portfolio\trigger.py:222:        last_tc = triggered_consensus.get(ticker, "HOLD")
portfolio\trigger.py:239:                triggered_consensus[ticker] = action
portfolio\trigger.py:243:            triggered_consensus[ticker] = action
portfolio\trigger.py:246:            triggered_consensus[ticker] = "HOLD"
portfolio\trigger.py:250:            triggered_consensus[ticker] = action
portfolio\trigger.py:251:    state["triggered_consensus"] = triggered_consensus
portfolio\trigger.py:348:        # classify_tier() can correctly detect the first real trigger of the day.
portfolio\trigger.py:419:    classify_tier() has already chosen T2 — T1 and T3 are never affected.
portfolio\trigger.py:432:def classify_tier(reasons, state=None):
portfolio\trigger.py:487:    Called by the main loop after classify_tier() to persist tier-specific state.
portfolio\claude_gate.py:8:Direct ``subprocess.Popen([claude_cmd, "-p", ...])`` calls are FORBIDDEN.
portfolio\claude_gate.py:258:def _find_claude_cmd() -> str | None:
portfolio\claude_gate.py:462:    claude_cmd = _find_claude_cmd()
portfolio\claude_gate.py:463:    if not claude_cmd:
portfolio\claude_gate.py:479:        claude_cmd, "-p", prompt,
portfolio\claude_gate.py:579:    claude_cmd = _find_claude_cmd()
portfolio\claude_gate.py:580:    if not claude_cmd:
portfolio\claude_gate.py:585:        claude_cmd, "-p", prompt,
portfolio\multi_agent_layer2.py:136:    claude_cmd = shutil.which("claude")
portfolio\multi_agent_layer2.py:137:    if not claude_cmd:
portfolio\multi_agent_layer2.py:161:            claude_cmd, "-p", prompt,
rg: tests: The system cannot find the file specified. (os error 2)

 succeeded in 600ms:
  580:     # - 2026-04-15: 360s. Telegram alerts at 10:34 showed recurring BUG-178
  581:     #   pool-timeout cycles across 2026-04-14/15 with the 5 zombie threads
  582:     #   completing 330-525s into the cycle, all 5 within ~10s of each
  583:     #   other — the signature of a shared-resource wait rather than truly
  584:     #   stuck work. Since 2026-04-09 the ticker path has grown (vix_term_-
  585:     #   structure, DXY intraday cross-asset, per-ticker signal gating,
  586:     #   fundamental correlation cluster, per-ticker directional accuracy,
  587:     #   ETH qwen3 gate) and the llama_server rotation (2026-04-10) means
  588:     #   signals occasionally pull stale/miss data under contention bursts.
  589:     #   The old 180s was measured when the system had 12 tickers; with 5
  590:     #   tickers and more per-ticker work the cost moved legitimately, not
  591:     #   because something is "stuck". 360s is 2.8x the observed p50-slow
  592:     #   (~130s) and 0.7x the observed p95-slow (~525s), leaving 240s of
  593:     #   margin inside the 600s cadence for post-cycle LLM batch, trigger
  594:     #   detection, journal, and telegram. Loop contract's own cycle_dur
  595:     #   check at 600s remains the catch-all for genuine hangs. Batch 1 of
  596:     #   this fix (phase-level instrumentation in signal_engine) and batch
  597:     #   2 (signal_utility TTL cache) ship together so we can see per-phase
  598:     #   timing in future slow cycles and the next bump decision is
  599:     #   grounded in data, not guesswork. See docs/plans/2026-04-15-bug178-
  600:     #   instrumentation-timeout.md for the full rationale.
  601:     #
  602:     # If cycles start creeping above ~360s again, the first place to look
  603:     # is the BUG-178 phase log dumped by the slow-cycle diagnostic below —
  604:     # acc_load, utility_overlay, weighted_consensus, penalties, linear_-
  605:     # factor, and consensus_gate are each tagged in portfolio.log so a
  606:     # real bottleneck is identifiable without guessing.
  607:     _TICKER_POOL_TIMEOUT = 360
  608:     # OR-I-001: avoid context manager — __exit__ calls shutdown(wait=True)
  609:     # which blocks the loop when threads hang past the timeout.
  610:     pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
  611:     futures = {
  612:         pool.submit(_process_ticker, name, source): name
  613:         for name, source in active_items
  614:     }
  615:     try:
  616:         for future in as_completed(futures, timeout=_TICKER_POOL_TIMEOUT):
  617:             name, result = future.result()
  618:             if result is not None:
  619:                 tf_data[name] = result["tfs"]
  620:                 prices_usd[name] = result["price"]
  621:                 signals[name] = {
  622:                     "action": result["action"],
  623:                     "confidence": result["confidence"],
  624:                     "indicators": result["ind"],
  625:                     "extra": result["extra"],
  626:                 }
  627:                 signals_ok += 1
  628:             else:
  629:                 signals_failed += 1
  630:     except TimeoutError:
  631:         timed_out = [n for f, n in futures.items() if not f.done()]
  632:         try:
  633:             from portfolio.signal_engine import get_last_signal as _get_last
  634:             from portfolio.signal_engine import get_phase_log as _get_phase_log
  635:             last_sigs = {n: _get_last(n) for n in timed_out}
  636:             # 2026-04-15: also dump per-ticker phase breakdown when the pool
  637:             # times out. This tells us WHICH post-dispatch phase
  638:             # (acc_load / utility_overlay / weighted_consensus / penalties /
  639:             # linear_factor / consensus_gate / regime_gate) burned the time,
  640:             # so we can target the real bottleneck instead of coarsely blaming
  641:             # __post_dispatch__.
  642:             phase_logs = {n: _get_phase_log(n) for n in timed_out}
  643:         except Exception:
  644:             last_sigs = {}
  645:             phase_logs = {}
  646:         logger.error(
  647:             "BUG-178: Ticker pool timeout after %ds. Stuck: %s. Last signals: %s",
  648:             _TICKER_POOL_TIMEOUT, timed_out, last_sigs,
  649:         )
  650:         for name, phases in phase_logs.items():
  651:             if phases:
  652:                 # Format as 'phase=dur_s' pairs, one ticker per line. Keep on
  653:                 # one log line per ticker so Windows Event Log / tail -f stays
  654:                 # readable when 5 tickers time out simultaneously.
  655:                 phase_str = " ".join(f"{p}={d:.1f}s" for p, d in phases)
  656:                 logger.error("BUG-178 phases [%s]: %s", name, phase_str)
  657:         for f in futures:
  658:             f.cancel()
  659:         signals_failed += len(timed_out)
  660:     finally:
  661:         pool.shutdown(wait=False, cancel_futures=True)
  662: 
  663:     # --- Post-cycle LLM batch flush ---
  664:     # Ministral/Qwen3/fingpt cache misses were enqueued during parallel
  665:     # ticker processing. Now flush them sequentially, grouped by model

 succeeded in 603ms:
    1: """Health monitoring for the finance-analyzer Layer 1 loop."""
    2: 
    3: import logging
    4: import threading
    5: import time
    6: from datetime import UTC, datetime
    7: from pathlib import Path
    8: 
    9: from portfolio.file_utils import atomic_write_json, last_jsonl_entry, load_json, load_jsonl_tail
   10: 
   11: logger = logging.getLogger(__name__)
   12: 
   13: DATA_DIR = Path(__file__).resolve().parent.parent / "data"
   14: HEALTH_FILE = DATA_DIR / "health_state.json"
   15: 
   16: # C10/H17: Protect all read-modify-write sequences in health.py.
   17: _health_lock = threading.Lock()
   18: 
   19: 
   20: def update_health(cycle_count: int, signals_ok: int, signals_failed: int,
   21:                   last_trigger_reason: str = None, error: str = None):
   22:     """Called at end of each Layer 1 cycle to update health state."""
   23:     with _health_lock:
   24:         state = load_health()
   25:         state["last_heartbeat"] = datetime.now(UTC).isoformat()
   26:         state["cycle_count"] = cycle_count
   27:         state["signals_ok"] = signals_ok
   28:         state["signals_failed"] = signals_failed
   29:         state["uptime_seconds"] = time.time() - state.get("start_time", time.time())
   30:         if last_trigger_reason:
   31:             state["last_trigger_reason"] = last_trigger_reason
   32:             state["last_trigger_time"] = datetime.now(UTC).isoformat()
   33:             # Cache the invocation timestamp so check_agent_silence() can avoid
   34:             # re-parsing invocations.jsonl on every call.
   35:             state["last_invocation_ts"] = state["last_trigger_time"]
   36:         if error:
   37:             state["errors"] = state.get("errors", [])[-19:] + [
   38:                 {"ts": datetime.now(UTC).isoformat(), "error": error}
   39:             ]
   40:             state["error_count"] = state.get("error_count", 0) + 1
   41:         atomic_write_json(HEALTH_FILE, state)
   42: 
   43: 
   44: def load_health() -> dict:
   45:     """Load current health state. Returns defaults if missing or corrupt."""
   46:     state = load_json(HEALTH_FILE)
   47:     if state is not None:
   48:         return state
   49:     return {"start_time": time.time(), "cycle_count": 0, "error_count": 0, "errors": []}
   50: 
   51: 
   52: def reset_session_start():
   53:     """Reset start_time to current time — call at loop startup.
   54: 
   55:     Prevents uptime_seconds from inheriting a stale start_time
   56:     from a previous session's health_state.json.
   57:     """
   58:     with _health_lock:
   59:         state = load_health()
   60:         state["start_time"] = time.time()
   61:         atomic_write_json(HEALTH_FILE, state)
   62: 
   63: 
   64: def heartbeat() -> None:
   65:     """Touch only the last_heartbeat timestamp.
   66: 
   67:     Called as a one-shot or periodically from a keepalive thread while
   68:     long-blocking work is in flight. Layer 2 invocation can block up to
   69:     600s (T2) or 900s (T3), but update_health() only runs at end-of-cycle
   70:     (AFTER Layer 2 returns). Without periodic touches the dashboard
   71:     /api/health endpoint flips fresh→stale every triggering cycle, which
   72:     is misleading: the loop is alive, just waiting on the subprocess.
   73: 
   74:     Other state (cycle_count, signals_ok/failed, errors) is left untouched
   75:     — those reflect the previously-completed cycle, still the most recent
   76:     ground truth. update_health() at end-of-cycle overwrites them with
   77:     this cycle's results.
   78: 
   79:     Failure-tolerant: callers wrap in try/except since this is a "nice to
   80:     have" hint and must never crash the loop. The atomic write means a
   81:     partial run leaves the prior file intact.
   82:     """
   83:     with _health_lock:
   84:         state = load_health()
   85:         state["last_heartbeat"] = datetime.now(UTC).isoformat()
   86:         atomic_write_json(HEALTH_FILE, state)
   87: 
   88: 
   89: # Keepalive default interval. The dashboard's stale gate fires at 300s
   90: # (check_staleness max_age_seconds=300), so 60s gives 5x headroom while

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\agent_invocation.py'; "'$ranges=@(@{s=610;e=790}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 594ms:
  610:         logger.warning("journal context failed: %s", e)
  611: 
  612:     # Perception gate: skip low-value invocations
  613:     try:
  614:         from portfolio.perception_gate import should_invoke as _should_invoke
  615:         should, gate_reason = _should_invoke(reasons, tier)
  616:         if not should:
  617:             logger.info("Perception gate skipped: %s", gate_reason)
  618:             _log_trigger(reasons, "skipped_gate", tier=tier)
  619:             return False
  620:     except Exception as e:
  621:         logger.warning("perception gate error (passing through): %s", e)
  622: 
  623:     # BUG-214: Drawdown circuit breaker — first-ever automated risk gate on
  624:     # the primary trading path. Advisory below _DRAWDOWN_BLOCK_PCT, hard-block
  625:     # above it. Respects user's high risk tolerance (memory/feedback_risk_tolerance.md).
  626:     #
  627:     # 2026-05-02 (adversarial review 05-01 P0-5): the bare `except Exception`
  628:     # used to swallow all errors and proceed. That meant a portfolio in
  629:     # 50%+ drawdown could continue trading if anything in the check threw
  630:     # (ImportError, IO error on portfolio_state.json mid-rename, KeyError
  631:     # on a missing dd dict key). The fail-safe direction for a circuit
  632:     # breaker is BLOCK on failure, not pass.
  633:     #
  634:     # New behavior:
  635:     # - Per-portfolio errors (file read, dict access) are tolerated for THAT
  636:     #   portfolio only — we still check the other portfolio.
  637:     # - A complete failure to even load the check (ImportError) is logged
  638:     #   ERROR + treated as block (fail-safe).
  639:     # - The narrow per-portfolio try/except still tolerates transient I/O,
  640:     #   so a missing portfolio_state.json mid-rename doesn't take the loop
  641:     #   down.
  642:     _drawdown_context = ""
  643:     try:
  644:         from portfolio.risk_management import check_drawdown
  645:     except Exception as e:
  646:         logger.error("DRAWDOWN BLOCK: check_drawdown unavailable (%s) — fail-safe block", e)
  647:         _log_trigger(reasons, "blocked_drawdown_unavailable", tier=tier)
  648:         return False
  649: 
  650:     for label, pf_path in [("Patient", PATIENT_PORTFOLIO), ("Bold", BOLD_PORTFOLIO)]:
  651:         if not pf_path.exists():
  652:             continue
  653:         try:
  654:             dd = check_drawdown(str(pf_path), max_drawdown_pct=_DRAWDOWN_WARN_PCT)
  655:             if dd["current_drawdown_pct"] > _DRAWDOWN_BLOCK_PCT:
  656:                 logger.error(
  657:                     "DRAWDOWN BLOCK: %s portfolio at %.1f%% drawdown (>%.0f%%) — skipping invocation",
  658:                     label, dd["current_drawdown_pct"], _DRAWDOWN_BLOCK_PCT,
  659:                 )
  660:                 _log_trigger(reasons, f"blocked_drawdown_{label.lower()}", tier=tier)
  661:                 return False
  662:             if dd["current_drawdown_pct"] > _DRAWDOWN_WARN_PCT:
  663:                 logger.warning(
  664:                     "DRAWDOWN WARNING: %s portfolio at %.1f%% drawdown (peak %.0f, current %.0f SEK)",
  665:                     label, dd["current_drawdown_pct"], dd["peak_value"], dd["current_value"],
  666:                 )
  667:             _drawdown_context += (
  668:                 f"\n[DRAWDOWN {label}] {dd['current_drawdown_pct']:.1f}% from peak "
  669:                 f"(peak={dd['peak_value']:.0f}, current={dd['current_value']:.0f} SEK)"
  670:             )
  671:         except Exception as e:
  672:             # Per-portfolio failure: log ERROR (not WARNING), but tolerate so
  673:             # the OTHER portfolio still gets checked. This keeps a transient IO
  674:             # error on one file from disabling the gate entirely. If BOTH
  675:             # portfolios fail, neither will set the block flag, and the
  676:             # invocation proceeds — by design, since blocking trading on a pure
  677:             # IO race that the next cycle will re-check is too aggressive.
  678:             logger.error(
  679:                 "DRAWDOWN check failed for %s portfolio (proceeding for this portfolio only): %s",
  680:                 label, e,
  681:             )
  682: 
  683:     # Adversarial review 05-01 P1-12 (2026-05-02): trade-guards pre-execution gate.
  684:     # `should_block_trade` was implemented in trade_guards.py for ARCH-29 but
  685:     # never imported by production code — only by tests. Wire it here so an
  686:     # invocation triggered by a ticker that is in cooldown for BOTH Patient
  687:     # and Bold gets short-circuited before the multi-agent / subprocess spawn
  688:     # (saves ~600s of T2 subprocess + Claude tokens for a decision that
  689:     # cannot be acted on).
  690:     #
  691:     # Semantics:
  692:     #   1. Pull the trade_guard_warnings already computed by reporting.py and
  693:     #      stored in agent_summary.json.
  694:     #   2. Build _guard_context for the prompt (advisory) — Layer 2 should
  695:     #      see active cooldowns/loss-streaks regardless of the gate decision.
  696:     #   3. Block ONLY when should_block_trade(...) is True AND the trigger
  697:     #      ticker is blocked for BOTH strategies. Single-strategy block
  698:     #      proceeds (the unblocked strategy can still trade).
  699:     #   4. Failure to load warnings (missing agent_summary, IO race) is
  700:     #      fail-OPEN — unlike drawdown, cooldowns are soft constraints and a
  701:     #      single missed gate cycle is not a safety risk.
  702:     _guard_context = ""
  703:     try:
  704:         guard_result = _load_guard_warnings()
  705:     except Exception as e:
  706:         logger.warning("trade-guards load failed (proceeding): %s", e)
  707:         guard_result = {"warnings": [], "summary": "load_failed"}
  708: 
  709:     if guard_result.get("warnings"):
  710:         _guard_context += f"\n[TRADE GUARDS] {guard_result.get('summary', '')}"
  711:         for w in guard_result["warnings"][:10]:  # cap context size
  712:             sev = w.get("severity", "?")
  713:             tkr = w.get("ticker") or w.get("details", {}).get("ticker", "?")
  714:             strat = w.get("strategy") or w.get("details", {}).get("strategy", "?")
  715:             msg = w.get("message", w.get("guard", "?"))
  716:             _guard_context += f"\n  [{sev.upper()}] {tkr}/{strat}: {msg}"
  717: 
  718:     try:
  719:         from portfolio.trade_guards import should_block_trade
  720:         if should_block_trade(guard_result):
  721:             # Determine the trigger ticker and check whether BOTH strategies
  722:             # are blocked on it. Anything else (single-strategy block, or
  723:             # block on a different ticker than the trigger) is advisory.
  724:             trigger_ticker = _extract_ticker(reasons)
  725:             blocked_strategies = {
  726:                 w.get("strategy") or w.get("details", {}).get("strategy")
  727:                 for w in guard_result["warnings"]
  728:                 if w.get("severity") == "block"
  729:                 and (
  730:                     w.get("ticker") == trigger_ticker
  731:                     or w.get("details", {}).get("ticker") == trigger_ticker
  732:                 )
  733:             }
  734:             blocked_strategies.discard(None)
  735:             if {"patient", "bold"}.issubset(blocked_strategies):
  736:                 logger.error(
  737:                     "TRADE GUARDS BLOCK: %s in cooldown for BOTH strategies — "
  738:                     "skipping invocation",
  739:                     trigger_ticker,
  740:                 )
  741:                 _log_trigger(reasons, "blocked_trade_guards", tier=tier)
  742:                 return False
  743:     except Exception as e:
  744:         # Import failures or shape mismatches must not derail the invocation.
  745:         logger.warning("trade-guards gate failed (proceeding): %s", e)
  746: 
  747:     # Multi-agent mode: parallel specialists + synthesis (Coordinator Mode pattern)
  748:     # Enabled via config.layer2.multi_agent = true, only for T2/T3
  749:     try:
  750:         config = _load_config()
  751:         multi_agent = config.get("layer2", {}).get("multi_agent", False)
  752:     except Exception:
  753:         multi_agent = False
  754: 
  755:     if multi_agent and tier >= 2:
  756:         try:
  757:             from portfolio.multi_agent_layer2 import (
  758:                 build_synthesis_prompt,
  759:                 launch_specialists,
  760:                 wait_for_specialists,
  761:             )
  762:             # Extract primary ticker from reasons
  763:             ticker = _extract_ticker(reasons)
  764:             logger.info("Multi-agent T%d: launching 3 specialists for %s", tier, ticker)
  765:             procs = launch_specialists(ticker, reasons)
  766:             if procs:
  767:                 # C3/NEW-1: timeout reduced from 150s to 30s (configurable via
  768:                 # layer2.specialist_timeout_s) to avoid blocking the main loop.
  769:                 # TODO: run specialists in background thread, collect results async.
  770:                 specialist_timeout = config.get("layer2", {}).get("specialist_timeout_s", 30)
  771:                 results = wait_for_specialists(procs, timeout=specialist_timeout)
  772:                 success_count = sum(1 for v in results.values() if v)
  773:                 logger.info("Specialists complete: %d/%d succeeded", success_count, len(results))
  774:                 # Even if some fail, proceed with synthesis using available reports
  775:                 prompt = build_synthesis_prompt(ticker, reasons)
  776:                 # Fall through to normal agent launch with synthesis prompt
  777:             else:
  778:                 logger.warning("No specialists launched, falling back to single-agent")
  779:                 prompt = _build_tier_prompt(tier, reasons)
  780:         except Exception as e:
  781:             logger.warning("Multi-agent failed (%s), falling back to single-agent", e)
  782:             prompt = _build_tier_prompt(tier, reasons)
  783:     else:
  784:         prompt = _build_tier_prompt(tier, reasons)
  785: 
  786:     # BUG-214: Append drawdown context so Layer 2 sees current risk levels.
  787:     if _drawdown_context:
  788:         prompt += "\n\n[RISK DATA]" + _drawdown_context
  789:     # P1-12 (2026-05-02): also surface trade-guard warnings to Layer 2 so
  790:     # it can avoid suggesting actions that the guards would just block in

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\agent_invocation.py'; "'$ranges=@(@{s=832;e=905}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 618ms:
  832:     log_fh = None
  833:     try:
  834:         agent_log_path = DATA_DIR / "agent.log"
  835:         # Capture the current file size BEFORE opening in append mode, so
  836:         # check_agent_completion() can read only this invocation's output
  837:         # (for auth-error detection) and not the entire log history.
  838:         global _agent_log_start_offset
  839:         _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
  840:         log_fh = open(agent_log_path, "a", encoding="utf-8")
  841:         # Strip Claude Code session markers to avoid "nested session" error
  842:         # when the parent process tree has Claude Code running
  843:         agent_env = os.environ.copy()
  844:         agent_env.pop("CLAUDECODE", None)
  845:         agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
  846:         # Increase Node.js stack size to prevent stack overflow in Claude CLI
  847:         agent_env["NODE_OPTIONS"] = "--stack-size=16384"
  848:         # P2 (2026-04-17): mark this subprocess as headless so CLAUDE.md's
  849:         # STARTUP CHECK protocol doesn't ask "How would you like to proceed?"
  850:         # when it finds unresolved critical_errors.jsonl entries. The agent
  851:         # has no stdin (pipe only), so any prompt that blocks on user input
  852:         # makes it hit the tier timeout with zero work done. The CLAUDE.md
  853:         # conditional turns that into "log the unresolved entries in your
  854:         # journal entry and proceed with the trigger task".
  855:         agent_env["PF_HEADLESS_AGENT"] = "1"
  856:         _agent_proc = subprocess.Popen(
  857:             cmd,
  858:             cwd=str(BASE_DIR),
  859:             stdout=log_fh,
  860:             stderr=subprocess.STDOUT,
  861:             env=agent_env,
  862:         )
  863:         _agent_log = log_fh  # transfer ownership on success
  864:         log_fh = None  # prevent cleanup below from closing it
  865:         _agent_start = time.monotonic()
  866:         _agent_start_wall = time.time()  # wall-clock fallback for P2B
  867:         _agent_timeout = timeout
  868:         _agent_tier = tier
  869:         _agent_reasons = list(reasons)
  870:         _journal_ts_before = _safe_last_jsonl_ts(JOURNAL_FILE, "journal")
  871:         _telegram_ts_before = _safe_last_jsonl_ts(TELEGRAM_FILE, "telegram")
  872:         # BUG-219: Snapshot transaction counts so check_agent_completion()
  873:         # can detect new trades and call record_trade().
  874:         global _patient_txn_count_before, _bold_txn_count_before
  875:         try:
  876:             from portfolio.file_utils import load_json
  877:             _patient_txn_count_before = len(
  878:                 (load_json(PATIENT_PORTFOLIO, default={}) or {}).get("transactions", [])
  879:             )
  880:             _bold_txn_count_before = len(
  881:                 (load_json(BOLD_PORTFOLIO, default={}) or {}).get("transactions", [])
  882:             )
  883:         except Exception:
  884:             _patient_txn_count_before = 0
  885:             _bold_txn_count_before = 0
  886:         # 2026-04-17: Publish the tier into health_state so loop_contract
  887:         # can pick the right per-tier grace window for the journal-activity
  888:         # check. Without this, the contract defaults to T3 grace (20m),
  889:         # which is conservative but can delay detection when an all-T1
  890:         # cadence runs silent. See loop_contract._get_layer2_grace_s() for
  891:         # the consumer and LAYER2_JOURNAL_GRACE_S_BY_TIER for the table.
  892:         # Best-effort: never fail the invocation because health_state is
  893:         # unwriteable (atomic_write_json handles the happy path; any
  894:         # exception is logged and swallowed).
  895:         try:
  896:             from portfolio.file_utils import atomic_write_json, load_json
  897:             # 2026-04-17 Codex P2: when claude is missing from PATH we fall
  898:             # back to pf-agent.bat which is unconditionally T3 regardless of
  899:             # the requested tier. Record the *effective* tier so the
  900:             # per-tier grace window in loop_contract reflects what's
  901:             # actually running.
  902:             effective_tier = 3 if not claude_cmd else tier
  903:             health_path = DATA_DIR / "health_state.json"
  904:             health = load_json(health_path, default={}) or {}
  905:             health["last_invocation_tier"] = effective_tier

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\agent_invocation.py'; "'$ranges=@(@{s=905;e=975}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 585ms:
  905:             health["last_invocation_tier"] = effective_tier
  906:             health["last_invocation_tier_ts"] = datetime.now(UTC).isoformat()
  907:             atomic_write_json(health_path, health)
  908:         except Exception as e:
  909:             logger.warning("health_state tier publish failed: %s", e)
  910:         logger.info(
  911:             "Agent T%d invoked pid=%s max_turns=%d timeout=%ds (%s)",
  912:             tier, _agent_proc.pid, max_turns, timeout,
  913:             ", ".join(reasons[:3]),
  914:         )
  915:         # 2026-05-05: arm the completion watchdog so the wall-clock
  916:         # timeout fires within ~30 s of the real budget even when the
  917:         # main loop's run() cycle bloats. See module-level note at
  918:         # _COMPLETION_WATCHDOG_INTERVAL_S.
  919:         _ensure_completion_watchdog()
  920:         # Save Layer 2 invocation notification (save-only, not sent to Telegram)
  921:         try:
  922:             config = _load_config()
  923:             reason_str = escape_markdown_v1(", ".join(reasons[:3]))
  924:             if len(reasons) > 3:
  925:                 reason_str += f" (+{len(reasons) - 3} more)"
  926:             tier_label = tier_cfg["label"]
  927:             notify_msg = f"_Layer 2 T{tier} ({tier_label}): {reason_str}_"
  928:             send_or_store(notify_msg, config, category="invocation")
  929:         except Exception as e:
  930:             logger.warning("invocation notification failed: %s", e)
  931:         return True
  932:     except Exception as e:
  933:         logger.error("invoking agent: %s", e)
  934:         if log_fh is not None:
  935:             log_fh.close()
  936:         return False
  937: 
  938: 
  939: def _write_fishing_context(journal_entry):
  940:     """Extract fishing context from Layer 2 journal entry.
  941: 
  942:     Called after Layer 2 completes. Creates a structured context file
  943:     that the fish engine reads as its strongest tactic vote.
  944:     """
  945:     try:
  946:         tickers = journal_entry.get('tickers', {})
  947:         xag = tickers.get('XAG-USD')
  948:         if not xag:
  949:             return
  950: 
  951:         outlook = xag.get('outlook', '')
  952:         conviction = float(xag.get('conviction', 0))
  953:         levels = xag.get('levels', [])
  954:         thesis = xag.get('thesis', '')
  955: 
  956:         # Determine direction bias
  957:         if outlook == 'bullish' and conviction >= 0.4:
  958:             direction_bias = 'bullish'
  959:             tactic_vote = 'LONG'
  960:             allow_long = True
  961:             allow_short = conviction < 0.6  # block short only if very bullish
  962:         elif outlook == 'bearish' and conviction >= 0.4:
  963:             direction_bias = 'bearish'
  964:             tactic_vote = 'SHORT'
  965:             allow_long = conviction < 0.6
  966:             allow_short = True
  967:         else:
  968:             direction_bias = 'neutral'
  969:             tactic_vote = None
  970:             allow_long = True
  971:             allow_short = True
  972: 
  973:         # Check for event context from watchlist
  974:         watchlist = journal_entry.get('watchlist', [])
  975:         event_context = ''

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"soft restart|restart|_startup_grace_active|check_triggers\\(|run\\(|while True|reload\\(|importlib|execv|spawn|KeyboardInterrupt|reinitialize|reset_session_start|main\\(\" portfolio" in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"invocations.jsonl|layer2_journal.jsonl|load_jsonl\\(|load_jsonl_tail|journal_index|tail\" portfolio" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 680ms:
portfolio\claude_gate.py:282:# A-IN-2 (2026-04-11): The previous code used `subprocess.run(timeout=...)`.
portfolio\claude_gate.py:283:# CPython's run() does kill the *direct* child on TimeoutExpired, but the
portfolio\claude_gate.py:284:# Claude CLI is a Node.js process that spawns its own helpers (MCP servers,
portfolio\claude_gate.py:288:# any local-LLM helpers Claude may have spawned.
portfolio\claude_gate.py:295:    """Return Popen kwargs that allow tree-killing the spawned process."""
portfolio\claude_gate.py:313:            res = subprocess.run(
portfolio\claude_gate.py:494:        # don't spawn 5 concurrent expensive Claude processes.
portfolio\trigger.py:65:# Startup grace period — after a restart, the first loop iteration updates the
portfolio\trigger.py:67:# every time the loop is restarted for a code update.
portfolio\trigger.py:69:_startup_grace_active = True  # True until first check_triggers call completes
portfolio\trigger.py:82:    false negatives. On process restart, monotonic origin resets and the
portfolio\trigger.py:84:    restart already resets the sustained counter).
portfolio\trigger.py:160:def check_triggers(signals, prices_usd, fear_greeds, sentiments):
portfolio\trigger.py:161:    global _startup_grace_active
portfolio\trigger.py:165:    # Startup grace period: on the first iteration after a restart, update the
portfolio\trigger.py:167:    # This lets the loop restart for code updates without spurious T3 reviews.
portfolio\trigger.py:170:    if _startup_grace_active and saved_pid != current_pid:
portfolio\trigger.py:197:        _startup_grace_active = False
portfolio\trigger.py:201:    _startup_grace_active = False
portfolio\trigger.py:379:# produced by check_triggers(). Reason shape stays stable across releases;
portfolio\main.py:310:    # for /api/accuracy so the first request after a dashboard restart
portfolio\main.py:426:def run(force_report=False, active_symbols=None):
portfolio\main.py:429:    # Check if a previously-spawned agent has completed (BUG-39).
portfolio\main.py:432:    # which polls every 30s independent of run()'s cadence. So when this
portfolio\main.py:807:    triggered, reasons = check_triggers(signals, prices_usd, fear_greeds, sentiments)
portfolio\main.py:984:    """Persist crash counter to survive process restarts."""
portfolio\main.py:1163:    from portfolio.health import reset_session_start
portfolio\main.py:1164:    reset_session_start()
portfolio\main.py:1188:        initial_report = run(force_report=True)
portfolio\main.py:1197:    except KeyboardInterrupt:
portfolio\main.py:1210:    while True:
portfolio\main.py:1223:            report = run(force_report=False, active_symbols=active_symbols)
portfolio\main.py:1226:        except KeyboardInterrupt:
portfolio\main.py:1393:        run(force_report="--report" in args)
portfolio\agent_invocation.py:68:# was called only once per ``main.run()`` cycle. When the cycle bloats
portfolio\agent_invocation.py:73:# the same check every 30s independent of ``run()``'s cadence; the
portfolio\agent_invocation.py:109:    Idempotent: the spawn happens at most once per process under normal
portfolio\agent_invocation.py:112:    failure or interpreter restart between calls could leave the global
portfolio\agent_invocation.py:113:    pointing at a dead thread), spawn a fresh one. Resets the stop
portfolio\agent_invocation.py:117:    Uses ``_completion_lock`` to make the is-alive-check + spawn atomic
portfolio\agent_invocation.py:118:    so concurrent callers cannot both pass the check and both spawn (a
portfolio\agent_invocation.py:163:    """Persist stack overflow counter to survive loop restarts."""
portfolio\agent_invocation.py:363:    (set alongside `_agent_start` at spawn) so we still recover a
portfolio\agent_invocation.py:473:        NOT spawn a replacement in that case because the old process
portfolio\agent_invocation.py:489:        result = subprocess.run(
portfolio\agent_invocation.py:586:                # BUG-92: If kill failed, don't spawn new agent (old one may
portfolio\agent_invocation.py:590:                        "Not spawning new agent — old process may still be running"
portfolio\agent_invocation.py:687:    # and Bold gets short-circuited before the multi-agent / subprocess spawn
portfolio\agent_invocation.py:917:        # main loop's run() cycle bloats. See module-level note at
portfolio\agent_invocation.py:1200:    # spawning the subprocess, so we only scan output from this invocation.
portfolio\health.py:52:def reset_session_start():
portfolio\health.py:142:    def _run(self) -> None:
portfolio\multi_agent_layer2.py:146:    # P2 follow-up (Codex P1 #1, 2026-04-17): specialists spawn as headless
portfolio\loop_contract.py:164:# respawns, the wall-clock gap trigger→journal can exceed the flat 18m
portfolio\loop_contract.py:234:    """Populated during run() to track what actually happened this cycle."""
portfolio\loop_contract.py:316:    # is demonstrably in flight. When T3 (900s) times out and respawns,
portfolio\loop_contract.py:350:    # skipped_stack_overflow     — pre-flight guard before spawning ✓
portfolio\loop_contract.py:1326:    Persists state to CONTRACT_STATE_FILE so escalation survives restarts.

 succeeded in 686ms:
portfolio\claude_gate.py:51:INVOCATIONS_LOG = DATA_DIR / "claude_invocations.jsonl"
portfolio\claude_gate.py:53:# session must see. Intentionally separate from claude_invocations.jsonl so
portfolio\claude_gate.py:275:    for entry in load_jsonl(INVOCATIONS_LOG):
portfolio\claude_gate.py:658:    entries = load_jsonl(INVOCATIONS_LOG)
portfolio\loop_contract.py:33:LAYER2_JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio\loop_contract.py:95:CLAUDE_INVOCATIONS_FILE = DATA_DIR / "claude_invocations.jsonl"
portfolio\loop_contract.py:101:LAYER2_INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
portfolio\loop_contract.py:181:    # ran 397-538s (see data/invocations.jsonl). The flat 180s grace was
portfolio\loop_contract.py:264:    details: dict = field(default_factory=dict)
portfolio\loop_contract.py:329:    # claude_invocations.jsonl. The latter is written by claude_gate for
portfolio\loop_contract.py:378:    # last_invocation_caller / status / ts in the alert details.
portfolio\loop_contract.py:383:    # fix-agent etc — its tail is essentially noise here, and it caused
portfolio\loop_contract.py:384:    # misleading violation rows on 2026-05-03/04 where details showed
portfolio\loop_contract.py:449:        details={
portfolio\loop_contract.py:485:                context=violation.details,
portfolio\loop_contract.py:523:            details={
portfolio\loop_contract.py:543:                details={
portfolio\loop_contract.py:567:            details={
portfolio\loop_contract.py:581:            details={
portfolio\loop_contract.py:593:            details={"flushed": False},
portfolio\loop_contract.py:616:            details={"invalid": invalid_signals},
portfolio\loop_contract.py:625:            details={"updated": False},
portfolio\loop_contract.py:634:            details={"written": False},
portfolio\loop_contract.py:663:                details={"dropped": dropped},
portfolio\loop_contract.py:672:            details={"updated": False},
portfolio\loop_contract.py:687:            details={"failed_tasks": failed_tasks},
portfolio\loop_contract.py:692:    # failed" pattern. See check_layer2_journal_activity() for details.
portfolio\loop_contract.py:795:            details={"path": str(ACCURACY_SNAPSHOTS_FILE)},
portfolio\loop_contract.py:820:        details={
portfolio\loop_contract.py:874:        details={
portfolio\loop_contract.py:972:            details=entry.get("context") or {},
portfolio\loop_contract.py:1067:                context=dict(v.details or {}),
portfolio\loop_contract.py:1147:            details={"missing": sorted(missing), "ok": sorted(report.underlying_tickers_ok)},
portfolio\loop_contract.py:1159:            details={
portfolio\loop_contract.py:1171:            details={"alive": False},
portfolio\loop_contract.py:1181:            details={"duration_s": duration, "limit_s": METALS_MAX_CYCLE_DURATION_S},
portfolio\loop_contract.py:1190:            details={"active_positions": report.active_positions},
portfolio\loop_contract.py:1199:            details={"reconciled": False},
portfolio\loop_contract.py:1208:            details={"errors": report.errors[:5]},
portfolio\loop_contract.py:1257:            details={"completed": False},
portfolio\loop_contract.py:1266:            details={"collected": False},
portfolio\loop_contract.py:1275:            details={"alive": False},
portfolio\loop_contract.py:1289:            details={
portfolio\loop_contract.py:1303:            details={"duration_s": duration, "limit_s": BOT_MAX_CYCLE_DURATION_S},
portfolio\loop_contract.py:1312:            details={"on_schedule": False},
portfolio\loop_contract.py:1370:                    details={**v.details, "consecutive": count},
portfolio\loop_contract.py:1434:        if v.details:
portfolio\loop_contract.py:1435:            for k, val in v.details.items():
portfolio\loop_contract.py:1466:            "details": v.details,
portfolio\loop_contract.py:1495:    details: dict | None,
portfolio\loop_contract.py:1504:    (Claude review of a85a646f, P1-1/P1-2: dashboard read ``details``
portfolio\loop_contract.py:1509:    ``details`` and applies the prefix strip before calling this helper.
portfolio\loop_contract.py:1513:      from ``details['alerts']`` because rendered messages embed
portfolio\loop_contract.py:1516:    - ``layer2_journal_activity`` folds ``details['trigger_time']`` so
portfolio\loop_contract.py:1530:    d = details if isinstance(details, dict) else {}
portfolio\loop_contract.py:1541:        # else fall through to message-only payload for legacy/empty-details
portfolio\loop_contract.py:1563:        violation.invariant, msg, violation.details
portfolio\autonomous.py:34:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio\autonomous.py:100:    prev_entries = load_jsonl(JOURNAL_FILE, limit=5)
portfolio\autonomous.py:719:        # those live in signal_details[]. Aggregate via weighted mean so the display
portfolio\autonomous.py:721:        details = p1d.get("signal_details", []) or []
portfolio\autonomous.py:723:        if details:
portfolio\autonomous.py:724:            weights = [float(d.get("weight", 0) or 0) for d in details]
portfolio\autonomous.py:725:            accs = [float(d.get("accuracy", 0) or 0) for d in details]
portfolio\agent_invocation.py:24:INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
portfolio\agent_invocation.py:25:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio\agent_invocation.py:71:# inflating ``duration_s`` in invocations.jsonl and delaying the kill
portfolio\agent_invocation.py:285:    Scans layer2_journal.jsonl (most-recent-first) for entries mentioning
portfolio\agent_invocation.py:293:        entries = load_jsonl(JOURNAL_FILE)
portfolio\agent_invocation.py:336:    Uses efficient tail-read via last_jsonl_entry() (reads last 4KB only).
portfolio\agent_invocation.py:713:            tkr = w.get("ticker") or w.get("details", {}).get("ticker", "?")
portfolio\agent_invocation.py:714:            strat = w.get("strategy") or w.get("details", {}).get("strategy", "?")
portfolio\agent_invocation.py:726:                w.get("strategy") or w.get("details", {}).get("strategy")
portfolio\agent_invocation.py:731:                    or w.get("details", {}).get("ticker") == trigger_ticker
portfolio\agent_invocation.py:1101:    invocations.jsonl row, the other returns ``None`` because
portfolio\agent_invocation.py:1354:    entries = load_jsonl(INVOCATIONS_FILE)
portfolio\main.py:360:    for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl"):
portfolio\main.py:571:    #   = ~375s tail. 500s was 2x that max.
portfolio\main.py:653:                # one log line per ticker so Windows Event Log / tail -f stays
portfolio\health.py:9:from portfolio.file_utils import atomic_write_json, last_jsonl_entry, load_json, load_jsonl_tail
portfolio\health.py:34:            # re-parsing invocations.jsonl on every call.
portfolio\health.py:176:    # Try cached timestamp from health_state first (avoids re-parsing invocations.jsonl)
portfolio\health.py:181:    # Fall back to parsing invocations.jsonl if health_state doesn't have the timestamp.
portfolio\health.py:183:        invocations_file = DATA_DIR / "invocations.jsonl"
portfolio\health.py:378:    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
portfolio\health.py:379:    entries = load_jsonl_tail(signal_log, max_entries=50)
portfolio\health.py:427:    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
portfolio\health.py:428:    entries = load_jsonl_tail(signal_log, max_entries=recent_entries)
portfolio\reflection.py:27:JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
portfolio\reflection.py:93:    entries = load_jsonl(JOURNAL_FILE, limit=100)
portfolio\reflection.py:154:    reflections = load_jsonl(REFLECTIONS_FILE)
portfolio\reflection.py:228:    reflections = load_jsonl(REFLECTIONS_FILE)
portfolio\reporting.py:31:# pattern instead of operators needing to grep log tails.
portfolio\reporting.py:151:        # Collect enhanced signal summaries (compact, no sub_signals detail in top-level)
portfolio\reporting.py:864:    Full detail (including per-signal _votes) only for "interesting" tickers:
portfolio\reporting.py:909:                    extra["_vote_detail"] = " | ".join(parts) if parts else "none"
portfolio\reporting.py:1167:    # Held positions with actionable detail
portfolio\reporting.py:1205:    full_detail_tickers = held_tickers | triggered_tickers
portfolio\reporting.py:1208:        if ticker not in full_detail_tickers:
portfolio\reporting.py:1233:        if ticker in full_detail_tickers:
portfolio\reporting.py:1234:            # Full detail — same as compact summary for held/interesting tickers
portfolio\reporting.py:1249:            # Medium detail — vote detail string + timeframes, no full _votes
portfolio\reporting.py:1262:                    extra["_vote_detail"] = " | ".join(parts) if parts else "none"

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"def load_jsonl\\(|def load_jsonl_tail\\(|def atomic_append_jsonl\\(|def last_jsonl_entry\\(\" portfolio\\file_utils.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\file_utils.py'; "'$ranges=@(@{s=120;e=240}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-10T15:27:02.888412Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:
rg: portfolio\file_utils.py: IO error for operation on portfolio\file_utils.py: The system cannot find the file specified. (os error 2)

 exited 1 in 564ms:
rg: portfolio\file_utils.py: IO error for operation on portfolio\file_utils.py: The system cannot find the file specified. (os error 2)

2026-05-10T15:27:02.997642Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:
[31;1mGet-Content: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m … utils.py'; $ranges=@(@{s=120;e=240}); $i=0; [36;1mGet-Content $path[0m | ForEa …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                                               ~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mCannot find path 'Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\file_utils.py' because it does not exist.[0m

 exited 1 in 669ms:
[31;1mGet-Content: [0m
[31;1m[36;1mLine |[0m
[31;1m[36;1m[36;1m   2 | [0m … utils.py'; $ranges=@(@{s=120;e=240}); $i=0; [36;1mGet-Content $path[0m | ForEa …[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m     | [31;1m                                               ~~~~~~~~~~~~~~~~~[0m
[31;1m[36;1m[36;1m[0m[36;1m[0m[36;1m[31;1m[31;1m[36;1m     | [31;1mCannot find path 'Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\file_utils.py' because it does not exist.[0m

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --files -g "*file_utils*"' in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-10T15:27:08.654726Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.7 seconds
Output:

 exited 1 in 665ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'rg --files | rg "file_utils|jsonl|health.py|trigger.py|agent_invocation.py|market_timing.py$"' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 629ms:
portfolio\trigger.py
portfolio\market_timing.py
portfolio\health.py
portfolio\agent_invocation.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"def load_jsonl\\(|def load_jsonl_tail\\(|def atomic_append_jsonl\\(|def last_jsonl_entry\\(\"" in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-10T15:27:19.574087Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:

 exited 1 in 616ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command 'Get-ChildItem -Name portfolio' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 535ms:
agent_invocation.py
autonomous.py
claude_gate.py
config_validator.py
health.py
loop_contract.py
main.py
market_timing.py
multi_agent_layer2.py
perception_gate.py
reflection.py
reporting.py
session_calendar.py
tickers.py
trigger.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\trigger.py'; "'$ranges=@(@{s=320;e=390}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 570ms:
  320:                 and last_stable != sent
  321:                 and sent != "neutral"
  322:                 and last_stable != "neutral"
  323:             ):
  324:                 reasons.append(
  325:                     f"{ticker} sentiment {last_stable}->{sent} (sustained)"
  326:                 )
  327:             stable_sent[ticker] = sent
  328:     state["sustained_sentiment"] = sustained_sent
  329:     state["stable_sentiment"] = stable_sent
  330: 
  331:     triggered = len(reasons) > 0
  332: 
  333:     if triggered:
  334:         state["last_trigger_time"] = time.time()
  335:         state["last"] = {
  336:             "signals": {
  337:                 t: {"action": s["action"], "confidence": s["confidence"]}
  338:                 for t, s in signals.items()
  339:             },
  340:             "prices": dict(prices_usd),
  341:             "fear_greeds": {
  342:                 t: fg if isinstance(fg, dict) else {} for t, fg in fear_greeds.items()
  343:             },
  344:             "sentiments": dict(sentiments),
  345:             "time": time.time(),
  346:         }
  347:         # C4/NEW-2: only update last_trigger_date when a real trigger fires, so that
  348:         # classify_tier() can correctly detect the first real trigger of the day.
  349:         state["last_trigger_date"] = _today_str()
  350: 
  351:     # Track today_date for other purposes
  352:     state["today_date"] = _today_str()
  353: 
  354:     state["sustained_counts"] = sustained
  355:     _save_state(state)
  356: 
  357:     return triggered, reasons
  358: 
  359: 
  360: # ---------------------------------------------------------------------------
  361: # Tier classification
  362: # ---------------------------------------------------------------------------
  363: 
  364: # Full review interval: 4h during market hours, 4h off-hours (T1 only)
  365: _FULL_REVIEW_MARKET_HOURS = 4
  366: _FULL_REVIEW_OFF_HOURS = 4  # Off-hours caps at T1, not T3
  367: 
  368: # Option P (2026-04-17): confidence-aware tier downshift.
  369: # When every reason in a T2 trigger is either a low-conviction consensus
  370: # crossing (<TIER_DOWNSHIFT_CONFIDENCE) or a fade flip (*->HOLD sustained),
  371: # downshift T2 -> T1 to save Claude token budget. T3 triggers (first-of-day,
  372: # F&G extreme, periodic full review) are NEVER downshifted. Sustained
  373: # direction flips (BUY<->SELL) and non-consensus triggers (post-trade, price
  374: # move, sentiment) block downshift. Setting this to 0.0 disables downshift
  375: # without code change.
  376: TIER_DOWNSHIFT_CONFIDENCE = 0.40
  377: 
  378: # Precompiled patterns for downshift eligibility analysis on reason strings
  379: # produced by check_triggers(). Reason shape stays stable across releases;
  380: # if the format ever changes, these miss -> downshift fails open (tier
  381: # stays T2, safe over-invocation rather than under-invocation).
  382: #
  383: # Word boundaries (\b) on "consensus" and "flipped" prevent substring
  384: # collisions — e.g. a hypothetical future reason containing "nonconsensus"
  385: # or "preflipped" would NOT accidentally match and trigger a downshift.
  386: # Current check_triggers has no such reasons, but anchoring is cheap
  387: # insurance against future regressions. Added 2026-04-17 after an
  388: # adversarial self-review surfaced the issue.
  389: _CONSENSUS_CONF_RE = re.compile(r'\bconsensus (?:BUY|SELL) \((\d+)%\)')
  390: _FADE_FLIP_RE = re.compile(r'\bflipped (?:BUY|SELL)->HOLD \(sustained\)')

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\agent_invocation.py'; "'$ranges=@(@{s=1085;e=1125}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 665ms:
 1085:                 logger.info(
 1086:                     "BUG-219: recorded %s %s %s pnl=%.2f%% for overtrading guards",
 1087:                     strategy, direction, ticker, pnl_pct or 0.0,
 1088:                 )
 1089:     except Exception as e:
 1090:         logger.warning("BUG-219: record_trade wiring failed: %s", e)
 1091: 
 1092: 
 1093: def check_agent_completion():
 1094:     """Check if a running agent has completed and log completion info.
 1095: 
 1096:     Thread-safe: serialised by ``_completion_lock`` so the main-loop call
 1097:     site (``portfolio.main.run``) and the 30 s daemon watchdog
 1098:     (``_completion_watchdog``) cannot race on ``_agent_proc`` /
 1099:     ``_agent_start`` state. Both call paths share the same lock; whichever
 1100:     reaches the lock first observes the completion and writes the
 1101:     invocations.jsonl row, the other returns ``None`` because
 1102:     ``_agent_proc`` is cleared at the end of the handler.
 1103: 
 1104:     Returns:
 1105:         dict with the following keys (None if no agent is running or the
 1106:         agent is still in progress and under its timeout):
 1107: 
 1108:         * ``status`` — "success", "incomplete", "failed", "auth_error",
 1109:           "timeout" (P1B, 2026-04-17), or "stack_overflow"
 1110:         * ``exit_code`` — int or None (None on timeout-kill path)
 1111:         * ``duration_s`` — float, always >= 0 (P2B clamp)
 1112:         * ``tier`` — int, the tier of the completed agent
 1113:         * ``reasons`` — list[str], the triggers for this invocation
 1114:         * ``journal_written`` — bool
 1115:         * ``telegram_sent`` — bool
 1116:         * ``completed_at`` — ISO-8601 UTC timestamp
 1117:     """
 1118:     with _completion_lock:
 1119:         return _check_agent_completion_locked()
 1120: 
 1121: 
 1122: def _check_agent_completion_locked():
 1123:     """Body of ``check_agent_completion``. The caller MUST hold
 1124:     ``_completion_lock``. Split out so the watchdog tick can call into
 1125:     the same code path without re-acquiring the lock recursively.

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\agent_invocation.py'; "'$ranges=@(@{s=340;e=445}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 643ms:
  340: 
  341: def _safe_last_jsonl_ts(path, label):
  342:     """Return the last JSONL timestamp without failing the invocation flow."""
  343:     try:
  344:         return _last_jsonl_ts(path)
  345:     except Exception as e:
  346:         logger.warning("%s baseline read failed: %s", label, e)
  347:         return None
  348: 
  349: 
  350: def _safe_elapsed_s():
  351:     """Return elapsed-since-invoke seconds, robust to a poisoned _agent_start.
  352: 
  353:     P2B (2026-04-17): yesterday's 2026-04-16T13:45:45 critical_errors.jsonl
  354:     entry had duration_s=-1776254571.5 (matches time.monotonic() - time.time()).
  355:     Indicates some historical path seeded _agent_start with an epoch
  356:     timestamp instead of a monotonic value. Clamping at the source +
  357:     logging a diagnostic keeps downstream consumers trustworthy and
  358:     surfaces the bug if it recurs.
  359: 
  360:     Codex P2 #2 follow-up (2026-04-17): a naive clamp-to-0 silently
  361:     disabled the P1B timeout path — `elapsed > _agent_timeout` can never
  362:     be true when elapsed is always 0. Fall back to `_agent_start_wall`
  363:     (set alongside `_agent_start` at spawn) so we still recover a
  364:     plausible elapsed and the hung-agent kill still fires. If both
  365:     clocks are corrupted, return 0 — that's the pre-existing failure
  366:     mode, not a worse state.
  367:     """
  368:     raw = time.monotonic() - _agent_start
  369:     if raw >= 0:
  370:         return raw
  371:     # Monotonic is poisoned — try the wall-clock fallback.
  372:     if _agent_start_wall > 0:
  373:         wall_elapsed = time.time() - _agent_start_wall
  374:         if wall_elapsed >= 0:
  375:             logger.warning(
  376:                 "BUG-P2B: monotonic elapsed negative (raw=%.1fs, "
  377:                 "_agent_start=%.1f); falling back to wall-clock "
  378:                 "(%.1fs since _agent_start_wall=%.1f). "
  379:                 "Indicates _agent_start was seeded with a non-monotonic value.",
  380:                 raw, _agent_start, wall_elapsed, _agent_start_wall,
  381:             )
  382:             return wall_elapsed
  383:     # Both clocks bad — clamp to 0 and warn loudly.
  384:     logger.warning(
  385:         "BUG-P2B: negative elapsed AND no wall-clock fallback "
  386:         "(raw=%.1fs, _agent_start=%.1f, _agent_start_wall=%.1f) — "
  387:         "clamping to 0. Timeout enforcement will not fire this cycle.",
  388:         raw, _agent_start, _agent_start_wall,
  389:     )
  390:     return 0.0
  391: 
  392: 
  393: def _scan_agent_log_for_auth_failure(label: str, extra_context: dict | None = None) -> bool:
  394:     """Scan the captured agent.log slice for claude-CLI auth-error markers.
  395: 
  396:     P1-3 (2026-05-02 last-followups): the timeout-kill path
  397:     (``_kill_overrun_agent``) used to forget the dead subprocess without
  398:     inspecting what it had printed. ``check_agent_completion()`` already
  399:     runs this scan on the happy path (line 956), so a hung agent that
  400:     printed "Not logged in" before getting stuck on a network retry
  401:     would surface as ``timeout`` (not ``auth_error``) and never land in
  402:     ``critical_errors.jsonl``. That asymmetry is the same class of silent
  403:     auth outage that the March-April 2026 incident exposed — the whole
  404:     point of the journal is to make that failure mode impossible to miss.
  405: 
  406:     Helper exists at module level so both call sites
  407:     (``check_agent_completion`` and ``_kill_overrun_agent``) stay in sync
  408:     if the scan logic ever needs to evolve.
  409: 
  410:     Returns True iff an auth-error marker was detected in the new slice.
  411:     Never raises — IO or decode failures are swallowed and logged so a
  412:     transient log-read problem cannot break the kill / completion paths.
  413: 
  414:     Args:
  415:         label: Caller identifier used in the auth-failure record (e.g.
  416:             ``"layer2_t2_timeout"``). Tier and trigger context are pulled
  417:             from the module-level ``_agent_tier`` / ``_agent_reasons``.
  418:         extra_context: Optional dict merged into the auth-failure record's
  419:             ``context`` field (e.g. ``{"exit_code": 0, "duration_s": 12.3}``
  420:             on the completion path). Tier/reasons are always included; this
  421:             is for caller-specific extras.
  422:     """
  423:     try:
  424:         agent_log_path = DATA_DIR / "agent.log"
  425:         if not agent_log_path.exists():
  426:             return False
  427:         with open(agent_log_path, "rb") as f:
  428:             f.seek(_agent_log_start_offset)
  429:             new_output = f.read().decode("utf-8", errors="replace")
  430:         ctx = {
  431:             "tier": _agent_tier,
  432:             "reasons": (_agent_reasons or [])[:5],
  433:         }
  434:         if extra_context:
  435:             ctx.update(extra_context)
  436:         return detect_auth_failure(
  437:             new_output,
  438:             caller=label,
  439:             context=ctx,
  440:         )
  441:     except Exception as e:
  442:         logger.warning("Auth-error scan of agent.log failed (%s): %s", label, e)
  443:         return False
  444: 
  445: 

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\main.py'; "'$ranges=@(@{s=420;e=455},@{s=830;e=900},@{s=1150;e=1245}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 588ms:
  420:                 report.post_cycle_results["log_rotation"] = False
  421: 
  422: 
  423: # --- Main orchestrator ---
  424: 
  425: 
  426: def run(force_report=False, active_symbols=None):
  427:     _ss._run_cycle_id += 1
  428: 
  429:     # Check if a previously-spawned agent has completed (BUG-39).
  430:     # 2026-05-05: this call is now lock-protected and shares the
  431:     # _completion_lock with portfolio.agent_invocation._completion_watchdog,
  432:     # which polls every 30s independent of run()'s cadence. So when this
  433:     # cycle bloats (cycle_duration violations 2026-05-01..04), the watchdog
  434:     # still observes subprocess completion and enforces the per-tier
  435:     # wall-clock timeout — see docs/plans/2026-05-05-l2-completion-watchdog.md.
  436:     try:
  437:         completion = check_agent_completion()
  438:         if completion:
  439:             logger.info(
  440:                 "Agent completed: status=%s tier=%s duration=%.1fs",
  441:                 completion.get("status"), completion.get("tier"),
  442:                 completion.get("duration_s", 0),
  443:             )
  444:     except Exception as e:
  445:         logger.warning("check_agent_completion failed: %s", e)
  446: 
  447:     config = _load_config()
  448:     state = load_state()
  449:     fx_rate = fetch_usd_sek()
  450: 
  451:     market_state, default_symbols, _ = get_market_state()
  452:     _ss._current_market_state = market_state
  453:     active = active_symbols or default_symbols
  454: 
  455:     skipped = set(SYMBOLS.keys()) - active
  830:         # 2026-05-04: Wrap long-blocking work (Layer 2 T2/T3 = 600-900s
  831:         # subprocess; autonomous fallback = bounded but not instant) in a
  832:         # heartbeat keepalive. update_health() (the normal heartbeat write)
  833:         # only runs at end-of-cycle, so without periodic ticks the
  834:         # dashboard /api/health flips stale 300s into any triggering cycle
  835:         # even though the loop is alive and waiting on Claude CLI.
  836:         # The context manager's __exit__ runs on exceptions too, so the
  837:         # daemon thread is always cleaned up. Skip-paths (NO_TELEGRAM,
  838:         # outside agent window) are NOT wrapped — they don't block.
  839:         from portfolio.health import heartbeat_keepalive
  840: 
  841:         layer2_cfg = config.get("layer2", {})
  842:         if os.environ.get("NO_TELEGRAM"):
  843:             logger.info("[NO_TELEGRAM] Skipping agent invocation")
  844:             _log_trigger(reasons_list, "skipped_test", tier=tier)
  845:         elif layer2_cfg.get("enabled", True):
  846:             if _is_agent_window():
  847:                 with heartbeat_keepalive():
  848:                     result = invoke_agent(reasons_list, tier=tier)
  849:                 _log_trigger(reasons_list, "invoked" if result else "skipped_busy", tier=tier)
  850:             else:
  851:                 logger.info("Layer 2: outside market window, skipping")
  852:                 _log_trigger(reasons_list, "skipped_offhours", tier=tier)
  853:         else:
  854:             logger.info("Layer 2 disabled — autonomous mode")
  855:             from portfolio.autonomous import autonomous_decision
  856:             with heartbeat_keepalive():
  857:                 autonomous_decision(
  858:                     config, signals, prices_usd, fx_rate, state,
  859:                     reasons_list, tf_data, tier, triggered_tickers,
  860:                 )
  861:             _log_trigger(reasons_list, "autonomous", tier=tier)
  862:     else:
  863:         write_agent_summary(signals, prices_usd, fx_rate, state, tf_data)
  864:         report.summary_written = True
  865:         logger.info("No trigger.")
  866: 
  867:     # Big Bet detection — can invoke a 30s Claude subprocess per qualifying
  868:     # candidate (portfolio/bigbet.py:invoke_layer2_eval), with no per-cycle
  869:     # cap. Wrapped in keepalive so heartbeat stays fresh across multi-minute
  870:     # bigbet sweeps that would otherwise re-trip the dashboard stale gate.
  871:     bigbet_cfg = config.get("bigbet", {})
  872:     if bigbet_cfg.get("enabled", False):
  873:         try:
  874:             from portfolio.bigbet import check_bigbet
  875:             from portfolio.health import heartbeat_keepalive
  876:             with heartbeat_keepalive():
  877:                 check_bigbet(signals, prices_usd, fx_rate, tf_data, config)
  878:         except Exception as e:
  879:             logger.warning("Big Bet check failed: %s", e)
  880: 
  881:     # ISKBETS monitoring — same shape: each qualifying ticker can fire a 30s
  882:     # Claude gate subprocess (portfolio/iskbets.py:invoke_layer2_gate). With
  883:     # 5 Tier-1 tickers configured the worst case is ~150s of subprocess work,
  884:     # well past the 300s heartbeat threshold when stacked with bigbet+L2.
  885:     try:
  886:         from portfolio.health import heartbeat_keepalive
  887:         from portfolio.iskbets import check_iskbets
  888:         with heartbeat_keepalive():
  889:             check_iskbets(signals, prices_usd, fx_rate, tf_data, config)
  890:     except Exception as e:
  891:         logger.warning("ISKBETS check failed: %s", e)
  892: 
  893:     # Avanza pending order confirmations
  894:     try:
  895:         from portfolio.avanza_orders import check_pending_orders
  896:         check_pending_orders(config)
  897:     except Exception as e:
  898:         logger.warning("Avanza order check failed: %s", e)
  899: 
  900:     # Periodic trade reflection
 1150:                 age_min = int(age_seconds // 60)
 1151:                 msg = f"_LOOP RESTARTED_ — previous heartbeat was {age_min}m ago. Possible crash."
 1152:                 logger.warning(msg)
 1153:                 try:
 1154:                     config = _load_config()
 1155:                     from portfolio.message_store import send_or_store
 1156:                     send_or_store(msg, config, category="error")
 1157:                 except Exception as e2:
 1158:                     logger.debug("Restart notification failed: %s", e2)
 1159:         except Exception as e:
 1160:             logger.warning("Failed to check heartbeat staleness: %s", e)
 1161: 
 1162:     # Reset session start_time so uptime_seconds is accurate for this session
 1163:     from portfolio.health import reset_session_start
 1164:     reset_session_start()
 1165: 
 1166:     logger.info("Loop started")
 1167: 
 1168:     # Load Alpha Vantage fundamentals cache from disk
 1169:     try:
 1170:         from portfolio.alpha_vantage import load_persistent_cache
 1171:         load_persistent_cache()
 1172:     except Exception as e:
 1173:         logger.warning("Failed to load fundamentals cache: %s", e)
 1174: 
 1175:     config = _load_config()
 1176:     logger.info("Starting loop with market-aware scheduling. Ctrl+C to stop.")
 1177: 
 1178:     try:
 1179:         from portfolio.iskbets import handle_command
 1180:         from portfolio.telegram_poller import TelegramPoller
 1181:         poller = TelegramPoller(config, on_command=handle_command)
 1182:         poller.start()
 1183:         logger.info("ISKBETS Telegram poller started")
 1184:     except Exception as e:
 1185:         logger.warning("ISKBETS poller failed to start: %s", e)
 1186: 
 1187:     try:
 1188:         initial_report = run(force_report=True)
 1189:         _run_post_cycle(config, report=initial_report)
 1190:         _reset_crash_counter()
 1191:         try:
 1192:             atomic_write_text(DATA_DIR / "heartbeat.txt", datetime.now(UTC).isoformat())
 1193:             if initial_report is not None:
 1194:                 initial_report.heartbeat_updated = True
 1195:         except Exception as e:
 1196:             logger.warning("Heartbeat write after initial run failed: %s", e)
 1197:     except KeyboardInterrupt:
 1198:         logger.info("Loop interrupted during initial run, shutting down cleanly")
 1199:         return
 1200:     except Exception as e:
 1201:         import traceback
 1202:         # OR-P1-2 (2026-05-02): wrap alert+sleep in _safe_crash_recovery so
 1203:         # an alert helper failure (disk full on crash counter, etc.) still
 1204:         # leaves a minimum backoff before the next try.
 1205:         logger.error("in initial run: %s", e)
 1206:         _safe_crash_recovery(traceback.format_exc())
 1207: 
 1208:     last_state = None
 1209:     last_cycle_started = time.monotonic()
 1210:     while True:
 1211:         market_state, active_symbols, sleep_interval = get_market_state()
 1212:         if interval:
 1213:             sleep_interval = interval
 1214:         if market_state != last_state:
 1215:             logger.info(
 1216:                 "Schedule: %s — %d instruments, %ds interval",
 1217:                 market_state, len(active_symbols), sleep_interval
 1218:             )
 1219:             last_state = market_state
 1220:         _sleep_for_next_cycle(last_cycle_started, sleep_interval)
 1221:         cycle_started = time.monotonic()
 1222:         try:
 1223:             report = run(force_report=False, active_symbols=active_symbols)
 1224:             _run_post_cycle(config, report=report)
 1225:             _reset_crash_counter()
 1226:         except KeyboardInterrupt:
 1227:             logger.info("Loop interrupted, shutting down cleanly")
 1228:             break
 1229:         except Exception as e:
 1230:             import traceback
 1231:             tb_text = traceback.format_exc()
 1232:             logger.error("in run: %s", e)
 1233:             try:
 1234:                 from portfolio.health import update_health
 1235:                 update_health(cycle_count=_ss._run_cycle_id, signals_ok=0, signals_failed=0,
 1236:                               error=str(e))
 1237:             except Exception as e2:
 1238:                 logger.warning("Health update after crash failed: %s", e2)
 1239:             # OR-P1-2 (2026-05-02): _safe_crash_recovery guarantees a
 1240:             # minimum sleep even if both _crash_alert and _crash_sleep fail
 1241:             # (e.g. disk full breaking the counter file). Without this floor,
 1242:             # the loop could spin tight on persistent failure since
 1243:             # _sleep_for_next_cycle takes 0s when elapsed > interval.
 1244:             _safe_crash_recovery(tb_text)
 1245:             report = None

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"TIER_CONFIG|timeout\" portfolio\\agent_invocation.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 517ms:
41:# for timeout enforcement when `_agent_start` (monotonic) gets poisoned.
42:# The clamp alone could silently disable the P1B T1 timeout check; this
47:_agent_timeout = 900  # per-invocation timeout (set from tier config)
67:# subprocess.poll() and enforces the per-tier wall-clock timeout, but it
94:        # Event.wait(timeout) returns True if the event was set during
95:        # the wait — ie shutdown. Returning False means the timeout
136:def _stop_completion_watchdog(timeout_s: float = 1.0) -> None:
141:    exit. ``timeout_s`` is intentionally short — the worst case is a
149:        _watchdog_thread.join(timeout=timeout_s)
174:TIER_CONFIG = {
175:    1: {"max_turns": 15, "timeout": 120, "label": "QUICK CHECK"},
176:    2: {"max_turns": 40, "timeout": 600, "label": "SIGNAL ANALYSIS"},
177:    3: {"max_turns": 40, "timeout": 900, "label": "FULL REVIEW"},
361:    disabled the P1B timeout path — `elapsed > _agent_timeout` can never
396:    P1-3 (2026-05-02 last-followups): the timeout-kill path
401:    would surface as ``timeout`` (not ``auth_error``) and never land in
416:            ``"layer2_t2_timeout"``). Tier and trigger context are pulled
450:    be called from ``check_agent_completion``. Previously the timeout
453:    invoked 16:04:58 with timeout=120s completed at 16:15:01 = 603s).
455:    Logs the trigger with status="timeout" and clears ``_agent_proc`` /
460:    so the silent-auth-failure detector covers the timeout path too — not
505:        _agent_proc.wait(timeout=15)  # BUG-189: 15s for Claude CLI Node.js teardown
525:    auth_label = f"layer2_t{_agent_tier}_timeout" if _agent_tier else "layer2_timeout"
531:        "timeout",
540:    global _agent_proc, _agent_log, _agent_start, _agent_start_wall, _agent_timeout
562:    tier_cfg = TIER_CONFIG.get(tier, TIER_CONFIG[3])
563:    timeout = tier_cfg["timeout"]
570:    # writing its "timeout" row — exactly the double-log the lock was added
578:            # can't cause a negative elapsed that silently skips the timeout.
580:            if elapsed > _agent_timeout:
767:                # C3/NEW-1: timeout reduced from 150s to 30s (configurable via
768:                # layer2.specialist_timeout_s) to avoid blocking the main loop.
770:                specialist_timeout = config.get("layer2", {}).get("specialist_timeout_s", 30)
771:                results = wait_for_specialists(procs, timeout=specialist_timeout)
852:        # makes it hit the tier timeout with zero work done. The CLAUDE.md
867:        _agent_timeout = timeout
911:            "Agent T%d invoked pid=%s max_turns=%d timeout=%ds (%s)",
912:            tier, _agent_proc.pid, max_turns, timeout,
916:        # timeout fires within ~30 s of the real budget even when the
1106:        agent is still in progress and under its timeout):
1109:          "timeout" (P1B, 2026-04-17), or "stack_overflow"
1110:        * ``exit_code`` — int or None (None on timeout-kill path)
1135:        # Still running. P1B (2026-04-17): enforce the wall-clock timeout
1138:        # no new triggers came through (yesterday: T1 timeout=120s ran
1142:        if _agent_timeout and elapsed > _agent_timeout:
1147:                "status": "timeout",
1203:    # ``_scan_agent_log_for_auth_failure`` so the timeout-kill path
1206:    # the same asymmetry the timeout path used to have.
1228:        # completion-path and timeout-path dicts have symmetric shape.
1342:        dict with keys: total, success, incomplete, failed, timeout,
1346:    Codex P2 #4 follow-up (2026-04-17): "timeout" and "auth_error" were
1347:    being dropped entirely by the status filter. Before P1B, timeouts
1349:    P1B check_agent_completion enforces timeout every cycle — these
1352:    completion_rate honest (timeouts count as failures for rate).
1361:    timeout = 0
1364:    tracked_statuses = ("success", "incomplete", "failed", "timeout", "auth_error")
1392:        elif entry_status == "timeout":
1393:            timeout += 1
1404:        "timeout": timeout,

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"TIER_CONFIG\\[|TIER_CONFIG =|timeout\\] =|timeout\\\": 0|layer2.*timeout\" portfolio" in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-10T15:28:12.915030Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.6 seconds
Output:

 exited 1 in 561ms:
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"agent\\.log|DATA_DIR / \\\"agent\\.log\\\"|agent_log_path\" portfolio" in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-10T15:28:23.486990Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
rg: regex parse error:
    (?:agent\.log|DATA_DIR / \)
    ^
error: unclosed group

 exited 1 in 518ms:
rg: regex parse error:
    (?:agent\.log|DATA_DIR / \)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n 'agent\\.log|agent_log_path' portfolio" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 515ms:
portfolio\loop_contract.py:447:            f"Check data/agent.log and data/critical_errors.jsonl."
portfolio\agent_invocation.py:38:_agent_log_start_offset = 0  # byte offset of agent.log at invoke time, for auth-error scan on completion
portfolio\agent_invocation.py:394:    """Scan the captured agent.log slice for claude-CLI auth-error markers.
portfolio\agent_invocation.py:424:        agent_log_path = DATA_DIR / "agent.log"
portfolio\agent_invocation.py:425:        if not agent_log_path.exists():
portfolio\agent_invocation.py:427:        with open(agent_log_path, "rb") as f:
portfolio\agent_invocation.py:442:        logger.warning("Auth-error scan of agent.log failed (%s): %s", label, e)
portfolio\agent_invocation.py:458:    P1-3 (2026-05-02 last-followups): also scans the captured agent.log
portfolio\agent_invocation.py:518:    # P1-3 (2026-05-02 last-followups): scan the captured agent.log slice
portfolio\agent_invocation.py:834:        agent_log_path = DATA_DIR / "agent.log"
portfolio\agent_invocation.py:839:        _agent_log_start_offset = agent_log_path.stat().st_size if agent_log_path.exists() else 0
portfolio\agent_invocation.py:840:        log_fh = open(agent_log_path, "a", encoding="utf-8")
portfolio\agent_invocation.py:1196:    # 2026-04-13: Scan agent.log for auth-error markers (see claude_gate.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\agent_invocation.py'; "'$ranges=@(@{s=145;e=180}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 545ms:
  145:     """
  146:     global _watchdog_thread
  147:     _watchdog_stop.set()
  148:     if _watchdog_thread is not None:
  149:         _watchdog_thread.join(timeout=timeout_s)
  150:     _watchdog_thread = None
  151: 
  152: 
  153: def _load_stack_overflow_counter() -> int:
  154:     """Load persisted stack overflow counter. Returns 0 if missing/corrupt."""
  155:     from portfolio.file_utils import load_json
  156:     data = load_json(_STACK_OVERFLOW_FILE)
  157:     if data and isinstance(data.get("count"), int):
  158:         return data["count"]
  159:     return 0
  160: 
  161: 
  162: def _save_stack_overflow_counter(count: int) -> None:
  163:     """Persist stack overflow counter to survive loop restarts."""
  164:     from portfolio.file_utils import atomic_write_json
  165:     atomic_write_json(_STACK_OVERFLOW_FILE, {
  166:         "count": count,
  167:         "updated": datetime.now(UTC).isoformat(),
  168:     })
  169: 
  170: 
  171: _consecutive_stack_overflows = _load_stack_overflow_counter()
  172: 
  173: # Per-tier configuration
  174: TIER_CONFIG = {
  175:     1: {"max_turns": 15, "timeout": 120, "label": "QUICK CHECK"},
  176:     2: {"max_turns": 40, "timeout": 600, "label": "SIGNAL ANALYSIS"},
  177:     3: {"max_turns": 40, "timeout": 900, "label": "FULL REVIEW"},
  178: }
  179: 
  180: 

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"BUG-178|ticker pool timeout|thread_name_prefix=\\\"ticker\\\"|alive threads|ThreadPoolExecutor|cancel\\(|shutdown\\(wait=False|orphan|zombie|ticker thread\" portfolio\\main.py portfolio\\loop_contract.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-10T15:29:11.802172Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.5 seconds
Output:
rg: regex parse error:
    (?:BUG-178|ticker pool timeout|thread_name_prefix=\)
    ^
error: unclosed group

 exited 1 in 529ms:
rg: regex parse error:
    (?:BUG-178|ticker pool timeout|thread_name_prefix=\)
    ^
error: unclosed group

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n 'BUG-178|ticker pool timeout|thread_name_prefix=\"ticker\"|alive threads|ThreadPoolExecutor|cancel\\(|shutdown\\(wait=False|orphan|zombie|ticker thread' portfolio\\main.py portfolio\\loop_contract.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 498ms:
portfolio\main.py:294:    # BUG-178/W15-W16 follow-up (2026-04-16): daily snapshot writer + the
portfolio\main.py:475:    from concurrent.futures import ThreadPoolExecutor, as_completed
portfolio\main.py:564:    # BUG-178: Add timeout to prevent indefinite hangs from stuck tickers.
portfolio\main.py:580:    # - 2026-04-15: 360s. Telegram alerts at 10:34 showed recurring BUG-178
portfolio\main.py:581:    #   pool-timeout cycles across 2026-04-14/15 with the 5 zombie threads
portfolio\main.py:603:    # is the BUG-178 phase log dumped by the slow-cycle diagnostic below —
portfolio\main.py:610:    pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker")
portfolio\main.py:647:            "BUG-178: Ticker pool timeout after %ds. Stuck: %s. Last signals: %s",
portfolio\main.py:656:                logger.error("BUG-178 phases [%s]: %s", name, phase_str)
portfolio\main.py:658:            f.cancel()
portfolio\main.py:661:        pool.shutdown(wait=False, cancel_futures=True)
portfolio\main.py:703:    # BUG-178 slow-cycle diagnostic (added 2026-04-10, diag/bug178-end-of-
portfolio\main.py:704:    # cycle-snapshot). The ticker pool BUG-178 handler already logs per-ticker
portfolio\main.py:724:            # BUG-178 handler's Last signals log line.
portfolio\loop_contract.py:695:    # 12. Signal accuracy degradation (BUG-178/W15-W16 follow-up, 2026-04-16).

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\main.py'; "'$ranges=@(@{s=560;e=607},@{s=695;e=735}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 576ms:
  560:             return name, None
  561: 
  562:     max_workers = max(1, min(len(active_items), 8))
  563: 
  564:     # BUG-178: Add timeout to prevent indefinite hangs from stuck tickers.
  565:     #
  566:     # Timeline:
  567:     # - original: 120s (assumed 60s cycle cadence)
  568:     # - 2026-04-09 (CPU fingpt daemon era): 500s — bumped because the CPU
  569:     #   fingpt daemon was serializing every ticker's sentiment behind its
  570:     #   own global lock, stretching per-ticker latency to ~75s × 5 tickers
  571:     #   = ~375s tail. 500s was 2x that max.
  572:     # - 2026-04-09 (post feat/fingpt-in-llmbatch): 180s. The fingpt daemon
  573:     #   was retired; fingpt moved to portfolio.llm_batch as a post-cycle
  574:     #   phase via llama_server full GPU offload. Per-ticker work no longer
  575:     #   serialized on fingpt. Live measurement after the merge showed
  576:     #   cycles dropping from ~472s to ~226s with 45s/ticker average.
  577:     #   180s = 4x the observed per-ticker average and 2x the target "slow"
  578:     #   cycle of 90s, a comfortable safety margin for genuinely stuck
  579:     #   tickers (network timeouts, yfinance blocking).
  580:     # - 2026-04-15: 360s. Telegram alerts at 10:34 showed recurring BUG-178
  581:     #   pool-timeout cycles across 2026-04-14/15 with the 5 zombie threads
  582:     #   completing 330-525s into the cycle, all 5 within ~10s of each
  583:     #   other — the signature of a shared-resource wait rather than truly
  584:     #   stuck work. Since 2026-04-09 the ticker path has grown (vix_term_-
  585:     #   structure, DXY intraday cross-asset, per-ticker signal gating,
  586:     #   fundamental correlation cluster, per-ticker directional accuracy,
  587:     #   ETH qwen3 gate) and the llama_server rotation (2026-04-10) means
  588:     #   signals occasionally pull stale/miss data under contention bursts.
  589:     #   The old 180s was measured when the system had 12 tickers; with 5
  590:     #   tickers and more per-ticker work the cost moved legitimately, not
  591:     #   because something is "stuck". 360s is 2.8x the observed p50-slow
  592:     #   (~130s) and 0.7x the observed p95-slow (~525s), leaving 240s of
  593:     #   margin inside the 600s cadence for post-cycle LLM batch, trigger
  594:     #   detection, journal, and telegram. Loop contract's own cycle_dur
  595:     #   check at 600s remains the catch-all for genuine hangs. Batch 1 of
  596:     #   this fix (phase-level instrumentation in signal_engine) and batch
  597:     #   2 (signal_utility TTL cache) ship together so we can see per-phase
  598:     #   timing in future slow cycles and the next bump decision is
  599:     #   grounded in data, not guesswork. See docs/plans/2026-04-15-bug178-
  600:     #   instrumentation-timeout.md for the full rationale.
  601:     #
  602:     # If cycles start creeping above ~360s again, the first place to look
  603:     # is the BUG-178 phase log dumped by the slow-cycle diagnostic below —
  604:     # acc_load, utility_overlay, weighted_consensus, penalties, linear_-
  605:     # factor, and consensus_gate are each tagged in portfolio.log so a
  606:     # real bottleneck is identifiable without guessing.
  607:     _TICKER_POOL_TIMEOUT = 360
  695: 
  696:     _run_elapsed = time.monotonic() - _run_start
  697:     logger.info(
  698:         "Signal loop done: %d OK, %d failed in %.1fs (%.1fs/ticker avg)",
  699:         signals_ok, signals_failed, _run_elapsed,
  700:         _run_elapsed / max(signals_ok + signals_failed, 1),
  701:     )
  702: 
  703:     # BUG-178 slow-cycle diagnostic (added 2026-04-10, diag/bug178-end-of-
  704:     # cycle-snapshot). The ticker pool BUG-178 handler already logs per-ticker
  705:     # last_signal state on its 180 s timeout, but cycles that stay under 180 s
  706:     # never fire the handler — so slow paths in the 120-180 s range hide from
  707:     # us. Fire a warning-level diagnostic when a cycle exceeds 120 s so we
  708:     # capture per-ticker phase state retrospectively.
  709:     #
  710:     # Each value in last_sigs is (sig_name, elapsed_since_set) where sig_name
  711:     # is one of: __pre_dispatch__ (hung in sentiment/fear_greed/LLM enqueue),
  712:     # a concrete enhanced signal name (hung in the dispatch loop on that one),
  713:     # or __post_dispatch__ (hung in accuracy_stats / consensus / per-ticker
  714:     # gating). The `elapsed_since_set` value is how long ago the tracker was
  715:     # updated — if the cycle total is 150 s but elapsed_since_set for a
  716:     # ticker is only 2 s, the slow code is AFTER the last-tracked marker;
  717:     # if elapsed_since_set is ~150 s, the thread was stuck at that marker.
  718:     if _run_elapsed > 120:
  719:         try:
  720:             from portfolio.signal_engine import get_last_signal as _get_last
  721:             from portfolio.signal_engine import get_phase_log as _get_phase_log
  722:             # Use signals.keys() because those are the tickers that successfully
  723:             # returned from the pool. Timed-out tickers are already named by the
  724:             # BUG-178 handler's Last signals log line.
  725:             last_sigs = {n: _get_last(n) for n in signals}
  726:             logger.warning(
  727:                 "Slow cycle diagnostic: %.1fs total, last signals tracked: %s",
  728:                 _run_elapsed, last_sigs,
  729:             )
  730:             # 2026-04-15: also dump the post-dispatch phase breakdown for each
  731:             # ticker that returned successfully. On a slow cycle the phase log
  732:             # reveals which named phase (acc_load, utility_overlay, weighted_-
  733:             # consensus, penalties, linear_factor, consensus_gate, regime_gate)
  734:             # burned the budget — otherwise we only see the aggregate and can't
  735:             # target the fix.

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "rg -n \"def _log_trigger\\(|atomic_append_jsonl\\(|INVOCATIONS_FILE|invocations.jsonl\" portfolio\\agent_invocation.py portfolio\\main.py portfolio\\trigger.py portfolio\\loop_contract.py" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 506ms:
portfolio\loop_contract.py:95:CLAUDE_INVOCATIONS_FILE = DATA_DIR / "claude_invocations.jsonl"
portfolio\loop_contract.py:101:LAYER2_INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
portfolio\loop_contract.py:181:    # ran 397-538s (see data/invocations.jsonl). The flat 180s grace was
portfolio\loop_contract.py:327:    # IMPORTANT (Codex P1 2026-04-17): use LAYER2_INVOCATIONS_FILE (the
portfolio\loop_contract.py:329:    # claude_invocations.jsonl. The latter is written by claude_gate for
portfolio\loop_contract.py:333:    latest_l2_inv = last_jsonl_entry(LAYER2_INVOCATIONS_FILE)
portfolio\loop_contract.py:380:    # Use LAYER2_INVOCATIONS_FILE (the L2-specific log already read above
portfolio\loop_contract.py:381:    # for the in-flight gate). The global CLAUDE_INVOCATIONS_FILE has
portfolio\loop_contract.py:425:    # 2026-05-04: latest_inv now points at LAYER2_INVOCATIONS_FILE which
portfolio\loop_contract.py:1460:        atomic_append_jsonl(CONTRACT_LOG_FILE, {
portfolio\main.py:120:    INVOCATIONS_FILE,
portfolio\main.py:360:    for name in ("invocations.jsonl", "layer2_journal.jsonl", "telegram_messages.jsonl"):
portfolio\agent_invocation.py:24:INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
portfolio\agent_invocation.py:71:# inflating ``duration_s`` in invocations.jsonl and delaying the kill
portfolio\agent_invocation.py:243:def _log_trigger(reasons, status, tier=None):
portfolio\agent_invocation.py:251:    atomic_append_jsonl(INVOCATIONS_FILE, entry)
portfolio\agent_invocation.py:1101:    invocations.jsonl row, the other returns ``None`` because
portfolio\agent_invocation.py:1249:        atomic_append_jsonl(INVOCATIONS_FILE, log_entry)
portfolio\agent_invocation.py:1354:    entries = load_jsonl(INVOCATIONS_FILE)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\agent_invocation.py'; "'$ranges=@(@{s=230;e=260}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 558ms:
  230:     import re
  231:     for r in reasons:
  232:         # Match patterns like XAG-USD, BTC-USD, ETH-USD
  233:         m = re.search(r'\b([A-Z]{2,5}-USD)\b', r)
  234:         if m:
  235:             return m.group(1)
  236:         # Match stock tickers like NVDA, PLTR
  237:         m = re.search(r'\b([A-Z]{2,5})\b(?:\s+flipped|\s+crossed|\s+broke)', r)
  238:         if m:
  239:             return m.group(1)
  240:     return "XAG-USD"  # default to silver
  241: 
  242: 
  243: def _log_trigger(reasons, status, tier=None):
  244:     entry = {
  245:         "ts": datetime.now(UTC).isoformat(),
  246:         "reasons": reasons,
  247:         "status": status,
  248:     }
  249:     if tier is not None:
  250:         entry["tier"] = tier
  251:     atomic_append_jsonl(INVOCATIONS_FILE, entry)
  252: 
  253: 
  254: def _load_guard_warnings():
  255:     """Read trade_guard_warnings from agent_summary.json.
  256: 
  257:     P1-12 (2026-05-02): the trade-guards pre-execution gate consumes the
  258:     warnings already computed by reporting.py and stored in agent_summary.
  259:     Reading them here (rather than recomputing) keeps the gate consistent
  260:     with what Layer 2 sees in its prompt context, and is much cheaper.

codex
A few of these claims hinge on helper behavior outside the cited lines. I’m checking import resolution and call sites now so I can separate real defects from line-number theater.
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import inspect, portfolio.file_utils as fu
print(fu.__file__)
for name in ['load_jsonl','load_jsonl_tail','atomic_write_json','atomic_append_jsonl','last_jsonl_entry']:
    obj = getattr(fu, name, None)
    print(name, inspect.signature(obj) if obj else None)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import inspect, portfolio.trigger as t
print(t.__file__)
print('today_str', inspect.getsource(t._today_str))
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
2026-05-10T15:30:07.403474Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'portfolio.file_utils'

 exited 1 in 775ms:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'portfolio.file_utils'

2026-05-10T15:30:07.423812Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.8 seconds
Output:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\trigger.py", line 25, in <module>
    from portfolio.file_utils import atomic_write_json, load_json
ModuleNotFoundError: No module named 'portfolio.file_utils'

 exited 1 in 795ms:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "Q:\finance-analyzer\.worktrees\adv-orchestration\portfolio\trigger.py", line 25, in <module>
    from portfolio.file_utils import atomic_write_json, load_json
ModuleNotFoundError: No module named 'portfolio.file_utils'

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "if (Test-Path 'Q:\\finance-analyzer\\portfolio\\file_utils.py') { Write-Output 'exists'; Get-Item 'Q:\\finance-analyzer\\portfolio\\file_utils.py' | Select-Object -ExpandProperty FullName } else { Write-Output 'missing' }" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 566ms:
exists
Q:\finance-analyzer\portfolio\file_utils.py

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\portfolio\\file_utils.py'; "'$ranges=@(@{s=1;e=220},@{s=220;e=380}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command "@'
import inspect, sys
sys.path.insert(0, r'Q:\\finance-analyzer')
import portfolio.file_utils as fu
print(fu.__file__)
for name in ['load_jsonl','load_jsonl_tail','atomic_write_json','atomic_append_jsonl','last_jsonl_entry']:
    obj = getattr(fu, name, None)
    print(name, inspect.signature(obj) if obj else None)
'@ | python -" in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 630ms:
    1: """Shared file I/O utilities."""
    2: import json
    3: import logging
    4: import os
    5: import tempfile
    6: from collections import deque
    7: from contextlib import suppress
    8: from pathlib import Path
    9: 
   10: # Cross-platform file-locking primitives for `atomic_append_jsonl`.
   11: # Same pattern as `portfolio/process_lock.py`.
   12: try:
   13:     import msvcrt as _msvcrt  # type: ignore[import-not-found]
   14: except ImportError:  # pragma: no cover - non-Windows
   15:     _msvcrt = None  # type: ignore[assignment]
   16: try:
   17:     import fcntl as _fcntl  # type: ignore[import-not-found]
   18: except ImportError:  # pragma: no cover - Windows
   19:     _fcntl = None  # type: ignore[assignment]
   20: 
   21: logger = logging.getLogger("portfolio.file_utils")
   22: 
   23: 
   24: def atomic_write_text(path, text, encoding="utf-8"):
   25:     """Atomically write text to a file using tempfile + os.replace.
   26: 
   27:     Same safety guarantees as atomic_write_json: fsync before replace,
   28:     no partial writes on crash.
   29:     """
   30:     path = Path(path)
   31:     path.parent.mkdir(parents=True, exist_ok=True)
   32:     fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
   33:     try:
   34:         with os.fdopen(fd, "w", encoding=encoding) as f:
   35:             f.write(text)
   36:             f.flush()
   37:             os.fsync(f.fileno())
   38:         os.replace(tmp, str(path))
   39:     except BaseException:
   40:         with suppress(OSError):
   41:             os.unlink(tmp)
   42:         raise
   43: 
   44: 
   45: def atomic_write_json(path, data, indent=2, ensure_ascii=True):
   46:     """Atomically write JSON data to a file using tempfile + os.replace.
   47: 
   48:     Ensures the file is never left in a partially-written state.
   49:     Fsyncs before replace to guarantee durability on power loss (H34).
   50:     """
   51:     path = Path(path)
   52:     path.parent.mkdir(parents=True, exist_ok=True)
   53:     fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
   54:     try:
   55:         with os.fdopen(fd, "w", encoding="utf-8") as f:
   56:             json.dump(data, f, indent=indent, default=str, ensure_ascii=ensure_ascii)
   57:             f.flush()
   58:             os.fsync(f.fileno())
   59:         os.replace(tmp, str(path))
   60:     except BaseException:
   61:         with suppress(OSError):
   62:             os.unlink(tmp)
   63:         raise
   64: 
   65: 
   66: def load_json(path, default=None):
   67:     """Load a JSON file. Returns *default* if missing or unparseable.
   68: 
   69:     Uses try/except instead of exists() check to avoid TOCTOU race.
   70:     Handles OSError (permission denied, locked files) gracefully on Windows.
   71:     Logs WARNING on corrupt JSON so corruption is observable (H35).
   72:     """
   73:     path = Path(path)
   74:     try:
   75:         return json.loads(path.read_text(encoding="utf-8"))
   76:     except FileNotFoundError:
   77:         return default
   78:     except OSError:
   79:         # BUG-139: PermissionError (file locked by antivirus/another process)
   80:         # and other OS-level errors should degrade gracefully like missing files.
   81:         logger.debug("load_json: OS error reading %s, returning default", path.name)
   82:         return default
   83:     except (json.JSONDecodeError, ValueError):
   84:         # H35: Log corruption so it's observable — silent defaults hide data loss.
   85:         logger.warning("load_json: corrupt JSON in %s, returning default", path.name)
   86:         return default
   87: 
   88: 
   89: def require_json(path):
   90:     """Load a JSON file, raising on corruption or missing file.
   91: 
   92:     Unlike load_json(), this function does NOT silently return defaults.
   93:     Use for critical files where corruption must be surfaced (H35).
   94: 
   95:     Raises:
   96:         FileNotFoundError: If the file does not exist.
   97:         json.JSONDecodeError: If the file contains invalid JSON.
   98:         OSError: If the file cannot be read.
   99:     """
  100:     path = Path(path)
  101:     return json.loads(path.read_text(encoding="utf-8"))
  102: 
  103: 
  104: def load_jsonl(path, limit=None):
  105:     """Load entries from a JSONL file.
  106: 
  107:     Args:
  108:         path: Path to the .jsonl file.
  109:         limit: If set, keep only the *last* N entries (uses a deque).
  110: 
  111:     Returns:
  112:         list of parsed dicts. Empty list if file missing or empty.
  113:     """
  114:     path = Path(path)
  115:     container = deque(maxlen=limit) if limit else []
  116:     try:
  117:         f = open(path, encoding="utf-8")
  118:     except FileNotFoundError:
  119:         return []
  120:     except OSError as e:
  121:         logger.warning("load_jsonl: cannot open %s: %s", path.name, e)
  122:         return []
  123:     with f:
  124:         for line in f:
  125:             line = line.strip()
  126:             if not line:
  127:                 continue
  128:             try:
  129:                 container.append(json.loads(line))
  130:             except json.JSONDecodeError as e:
  131:                 logger.debug("Skipping malformed JSONL line in %s: %s", path.name, str(e)[:100])
  132:                 continue
  133:     return list(container)
  134: 
  135: 
  136: def load_jsonl_tail(path, max_entries=500, tail_bytes=512_000):
  137:     """Load the last N entries from a JSONL file by reading from the end.
  138: 
  139:     Much more efficient than load_jsonl(limit=N) for large files because
  140:     it only reads the last `tail_bytes` bytes instead of the entire file.
  141: 
  142:     Args:
  143:         path: Path to the .jsonl file.
  144:         max_entries: Maximum entries to return.
  145:         tail_bytes: How many bytes to read from the end of the file.
  146:             Default 512KB ≈ ~1000 typical entries.
  147: 
  148:     Returns:
  149:         list of parsed dicts (chronological order). Empty list if missing.
  150:     """
  151:     path = Path(path)
  152:     try:
  153:         file_size = path.stat().st_size
  154:     except (FileNotFoundError, OSError):
  155:         return []
  156:     if file_size == 0:
  157:         return []
  158: 
  159:     entries = []
  160:     try:
  161:         with open(path, "rb") as f:
  162:             # Seek to near end of file
  163:             offset = max(0, file_size - tail_bytes)
  164:             # 2026-05-04 codex P3-1 follow-up: peek the byte just before
  165:             # the seek point. If it's a newline, the seek lands exactly
  166:             # at a line boundary and the first decoded line is intact.
  167:             # Without this check, a happy-coincidence boundary would
  168:             # cost us one valid entry on every read.
  169:             seek_on_boundary = False
  170:             if offset > 0:
  171:                 f.seek(offset - 1)
  172:                 prior = f.read(1)
  173:                 seek_on_boundary = prior == b"\n"
  174:             f.seek(offset)
  175:             data = f.read()
  176:         # Decode and split into lines
  177:         text = data.decode("utf-8", errors="replace")
  178:         lines = text.split("\n")
  179:         # Drop the first line only when we landed mid-line. When seek
  180:         # lands on a newline boundary, the first decoded line is
  181:         # complete and should be kept.
  182:         if offset > 0 and lines and not seek_on_boundary:
  183:             lines = lines[1:]
  184:         for line in lines:
  185:             line = line.strip()
  186:             if not line:
  187:                 continue
  188:             try:
  189:                 entries.append(json.loads(line))
  190:             except json.JSONDecodeError:
  191:                 continue
  192:     except (OSError, UnicodeDecodeError) as e:
  193:         logger.debug("load_jsonl_tail failed for %s: %s", path.name, e)
  194:         return []
  195: 
  196:     # Return last max_entries in chronological order
  197:     if len(entries) > max_entries:
  198:         entries = entries[-max_entries:]
  199:     return entries
  200: 
  201: 
  202: def atomic_append_jsonl(path, entry):
  203:     """Append a single JSON entry to a JSONL file with atomic semantics
  204:     across threads and processes.
  205: 
  206:     Implementation: binary-append (``"ab"``) to the target + an
  207:     exclusive lock on a *sidecar* lockfile held for the duration of
  208:     the ``write + flush + fsync`` sequence. Windows CRT does not
  209:     guarantee ``O_APPEND`` atomicity (unlike POSIX), so without a lock
  210:     heavy thread contention can produce torn lines (head bytes lost,
  211:     tail bytes survive).
  212: 
  213:     Sidecar-lockfile pattern (``<path>.lock``) — not the target file
  214:     itself — guarantees a non-empty, lockable byte-range exists even
  215:     when the target file is brand-new / size 0. This closes the race
  216:     window Codex flagged on 2026-04-17: two first-writers opening
  217:     the freshly-created target simultaneously could both have
  218:     failed the empty-file ``msvcrt.locking(fd, LK_LOCK, 1)`` call and
  219:     interleaved their writes.
  220: 
  221:     This primitive is used by ~20 JSONL writers across the codebase
  222:     (signal_log, claude_invocations, critical_errors, telegram_messages,
  223:     accuracy_snapshots, etc.) so the fix eliminates torn-line risk
  224:     system-wide. Unxfails
  225:     ``tests/test_fix_agent_dispatcher.py::test_concurrent_append_does_not_corrupt_jsonl``.
  226:     """
  227:     path = Path(path)
  228:     path.parent.mkdir(parents=True, exist_ok=True)
  229:     data = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
  230: 
  231:     # Sidecar lockfile — always non-empty so locking never fails on
  232:     # size-0 targets. Pre-create if missing; single byte is enough.
  233:     lock_path = path.parent / f".{path.name}.lock"
  234:     if not lock_path.exists():
  235:         try:
  236:             with open(lock_path, "ab") as lf:
  237:                 if lf.tell() == 0:
  238:                     lf.write(b"\0")
  239:         except OSError:
  240:             pass  # best-effort; lock open below will retry
  241: 
  242:     with open(lock_path, "rb+") as lock_f:
  243:         lfd = lock_f.fileno()
  244:         win_locked = False
  245:         try:
  246:             if _msvcrt is not None:
  247:                 os.lseek(lfd, 0, os.SEEK_SET)
  248:                 _msvcrt.locking(lfd, _msvcrt.LK_LOCK, 1)  # blocking
  249:                 win_locked = True
  250:             elif _fcntl is not None:
  251:                 _fcntl.flock(lfd, _fcntl.LOCK_EX)
  252:             with open(path, "ab") as f:
  253:                 f.write(data)
  254:                 f.flush()
  255:                 os.fsync(f.fileno())
  256:         finally:
  257:             if win_locked and _msvcrt is not None:
  258:                 try:
  259:                     os.lseek(lfd, 0, os.SEEK_SET)
  260:                     _msvcrt.locking(lfd, _msvcrt.LK_UNLCK, 1)
  261:                 except OSError:
  262:                     pass
  263:             # fcntl.flock releases automatically on close.
  264: 
  265: 
  266: def atomic_write_jsonl(path, entries):
  267:     """Atomically rewrite a JSONL file with the given entries.
  268: 
  269:     Uses tempfile + os.replace so the file is never left partially written.
  270:     """
  271:     path = Path(path)
  272:     path.parent.mkdir(parents=True, exist_ok=True)
  273:     fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
  274:     try:
  275:         with os.fdopen(fd, "w", encoding="utf-8") as f:
  276:             for entry in entries:
  277:                 f.write(json.dumps(entry, ensure_ascii=False) + "\n")
  278:             f.flush()
  279:             os.fsync(f.fileno())
  280:         os.replace(tmp, str(path))
  281:     except BaseException:
  282:         with suppress(OSError):
  283:             os.unlink(tmp)
  284:         raise
  285: 
  286: 
  287: def last_jsonl_entry(path, field=None):
  288:     """Return the last parsed JSON entry from a JSONL file (efficient tail read).
  289: 
  290:     Reads only the last 4KB of the file instead of scanning the entire file.
  291: 
  292:     Args:
  293:         path: Path to the JSONL file.
  294:         field: If set, return only this field's value from the last entry.
  295: 
  296:     Returns:
  297:         The last entry (dict) or the value of *field*, or None if file is
  298:         missing/empty/unreadable.
  299:     """
  300:     path = Path(path)
  301:     try:
  302:         file_size = path.stat().st_size
  303:     except (OSError, FileNotFoundError):
  304:         return None
  305:     if file_size == 0:
  306:         return None
  307:     read_size = min(file_size, 4096)
  308:     try:
  309:         with open(path, "rb") as f:
  310:             f.seek(max(0, file_size - read_size))
  311:             tail = f.read().decode("utf-8", errors="replace")
  312:     except OSError:
  313:         return None
  314:     for line in reversed(tail.strip().splitlines()):
  315:         line = line.strip()
  316:         if not line:
  317:             continue
  318:         try:
  319:             entry = json.loads(line)
  320:             if field is not None:
  321:                 return entry.get(field)
  322:             return entry
  323:         except (json.JSONDecodeError, AttributeError):
  324:             continue
  325:     return None
  326: 
  327: 
  328: def prune_jsonl(path, max_entries=5000):
  329:     """Prune a JSONL file to keep only the most recent *max_entries*.
  330: 
  331:     Reads the file, keeps the tail, and atomically rewrites it.
  332:     Skips malformed lines (e.g., from partial writes) during read.
  333:     No-op if the file has fewer entries than *max_entries*.
  334: 
  335:     Returns the number of entries removed, or 0 if no pruning was needed.
  336:     """
  337:     path = Path(path)
  338:     lines = []
  339:     try:
  340:         f = open(path, encoding="utf-8")
  341:     except FileNotFoundError:
  342:         return 0
  343:     with f:
  344:         for line in f:
  345:             stripped = line.strip()
  346:             if not stripped:
  347:                 continue
  348:             # Validate JSON to avoid preserving corrupt partial lines
  349:             try:
  350:                 json.loads(stripped)
  351:                 lines.append(stripped)
  352:             except json.JSONDecodeError:
  353:                 logger.warning("prune_jsonl: skipping malformed line in %s", path.name)
  354:     if len(lines) <= max_entries:
  355:         return 0
  356:     removed = len(lines) - max_entries
  357:     keep = lines[-max_entries:]
  358:     # Atomic rewrite via tempfile
  359:     fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
  360:     try:
  361:         with os.fdopen(fd, "w", encoding="utf-8") as f:
  362:             for line in keep:
  363:                 f.write(line + "\n")
  364:             f.flush()
  365:             os.fsync(f.fileno())
  366:         os.replace(tmp, str(path))
  367:     except BaseException:
  368:         with suppress(OSError):
  369:             os.unlink(tmp)
  370:         raise
  371:     logger.info("Pruned %s: removed %d entries, kept %d", path.name, removed, max_entries)
  372:     return removed

 succeeded in 651ms:
Q:\finance-analyzer\portfolio\file_utils.py
load_jsonl (path, limit=None)
load_jsonl_tail (path, max_entries=500, tail_bytes=512000)
atomic_write_json (path, data, indent=2, ensure_ascii=True)
atomic_append_jsonl (path, entry)
last_jsonl_entry (path, field=None)

exec
"C:\\Program Files\\PowerShell\\7\\pwsh.exe" -Command '$path='"'Q:\\finance-analyzer\\.worktrees\\adv-orchestration\\portfolio\\claude_gate.py'; "'$ranges=@(@{s=1;e=40},@{s=250;e=330},@{s=450;e=520}); $i=0; Get-Content $path | ForEach-Object { $i++; foreach($r in $ranges){ if($i -ge $r.s -and $i -le $r.e){ '"'{0,5}: {1}' -f "'$i, $_; break } } }' in Q:\finance-analyzer\.worktrees\adv-orchestration
 succeeded in 634ms:
    1: """Centralized Claude Code invocation gatekeeper.
    2: 
    3: This module is the ONLY approved way to invoke Claude Code (``claude -p``)
    4: from anywhere in the codebase.  All callers — agent_invocation, metals_loop,
    5: silver_monitor, claude_fundamental, analyze, bigbet, iskbets, etc. — MUST
    6: route through ``invoke_claude()`` defined here.
    7: 
    8: Direct ``subprocess.Popen([claude_cmd, "-p", ...])`` calls are FORBIDDEN.
    9: Doing so bypasses the kill switch, rate limiter, and invocation tracking.
   10: 
   11: Usage::
   12: 
   13:     from portfolio.claude_gate import invoke_claude
   14: 
   15:     success, exit_code = invoke_claude(
   16:         prompt="Analyze BTC-USD",
   17:         caller="silver_monitor",
   18:         model="sonnet",
   19:         max_turns=20,
   20:         timeout=180,
   21:     )
   22: """
   23: 
   24: import contextlib
   25: import json
   26: import logging
   27: import os
   28: import platform
   29: import shutil
   30: import signal
   31: import subprocess
   32: import time
   33: from datetime import UTC, datetime
   34: from pathlib import Path
   35: 
   36: from portfolio.file_utils import atomic_append_jsonl, load_jsonl
   37: 
   38: logger = logging.getLogger("portfolio.claude_gate")
   39: 
   40: import threading
  250:                     f"ANTHROPIC_API_KEY, or expired ~/.claude/.credentials.json."
  251:                 ),
  252:                 context={**(context or {}), "marker": marker},
  253:             )
  254:             return True
  255:     return False
  256: 
  257: 
  258: def _find_claude_cmd() -> str | None:
  259:     """Locate the ``claude`` CLI executable on PATH."""
  260:     return shutil.which("claude")
  261: 
  262: 
  263: def _log_invocation(entry: dict) -> None:
  264:     """Append an invocation record to the JSONL log."""
  265:     try:
  266:         atomic_append_jsonl(INVOCATIONS_LOG, entry)
  267:     except Exception as e:
  268:         logger.warning("Failed to write invocation log: %s", e)
  269: 
  270: 
  271: def _count_today_invocations() -> int:
  272:     """Count invocation records from today (UTC)."""
  273:     today_str = datetime.now(UTC).strftime("%Y-%m-%d")
  274:     count = 0
  275:     for entry in load_jsonl(INVOCATIONS_LOG):
  276:         ts = entry.get("timestamp", "")
  277:         if ts.startswith(today_str):
  278:             count += 1
  279:     return count
  280: 
  281: 
  282: # A-IN-2 (2026-04-11): The previous code used `subprocess.run(timeout=...)`.
  283: # CPython's run() does kill the *direct* child on TimeoutExpired, but the
  284: # Claude CLI is a Node.js process that spawns its own helpers (MCP servers,
  285: # the actual claude API client process, etc.). Killing the direct child
  286: # leaves all of its descendants running as zombies on Windows. Over a long
  287: # session this leaks file handles, sockets, and (worst) GPU VRAM held by
  288: # any local-LLM helpers Claude may have spawned.
  289: #
  290: # Fix: explicitly Popen with a new process group/session so we can kill the
  291: # entire tree, not just the direct child. On Windows we use taskkill /T /F
  292: # (kills the whole tree by PID); on Unix we use os.killpg(SIGKILL) on the
  293: # process group started via start_new_session=True.
  294: def _popen_kwargs_for_tree_kill() -> dict:
  295:     """Return Popen kwargs that allow tree-killing the spawned process."""
  296:     if platform.system() == "Windows":
  297:         return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
  298:     return {"start_new_session": True}
  299: 
  300: 
  301: def _kill_process_tree(proc: subprocess.Popen, *, label: str = "claude") -> None:
  302:     """Kill a Popen process and all of its descendants. Best-effort:
  303:     falls back to proc.kill() if the platform-specific path fails.
  304:     Always returns; never raises."""
  305:     if proc.poll() is not None:
  306:         return  # already exited
  307:     pid = proc.pid
  308:     try:
  309:         if platform.system() == "Windows":
  310:             # taskkill /T = terminate this PID and all child processes,
  311:             # /F = force, /PID = the parent PID. Capture stderr to keep
  312:             # logs clean if the process already exited between poll() and here.
  313:             res = subprocess.run(
  314:                 ["taskkill", "/T", "/F", "/PID", str(pid)],
  315:                 capture_output=True, timeout=5,
  316:             )
  317:             if res.returncode not in (0, 128):  # 128 = "process not found"
  318:                 logger.warning(
  319:                     "%s tree kill via taskkill returned %d (stderr=%r) — "
  320:                     "falling back to proc.kill()",
  321:                     label, res.returncode, res.stderr.decode("utf-8", "replace")[:200],
  322:                 )
  323:                 proc.kill()
  324:         else:
  325:             try:
  326:                 pgid = os.getpgid(pid)
  327:                 os.killpg(pgid, signal.SIGKILL)
  328:             except (ProcessLookupError, OSError) as e:
  329:                 logger.warning("%s killpg(%d) failed: %s — falling back to proc.kill()", label, pid, e)
  330:                 proc.kill()
  450:         })
  451:         return False, -1
  452: 
  453:     # --- Rate-limit warning ---
  454:     today_count = _count_today_invocations()
  455:     if today_count >= _DAILY_WARN_THRESHOLD:
  456:         logger.warning(
  457:             "Daily invocation count (%d) exceeds threshold (%d) — caller=%s",
  458:             today_count, _DAILY_WARN_THRESHOLD, caller,
  459:         )
  460: 
  461:     # --- Locate claude CLI ---
  462:     claude_cmd = _find_claude_cmd()
  463:     if not claude_cmd:
  464:         logger.error("claude CLI not found on PATH — caller=%s", caller)
  465:         _log_invocation({
  466:             "timestamp": now_iso,
  467:             "caller": caller,
  468:             "status": "error",
  469:             "reason": "claude not on PATH",
  470:             "model": model,
  471:             "max_turns": max_turns,
  472:             "duration_seconds": 0,
  473:             "exit_code": -1,
  474:         })
  475:         return False, -1
  476: 
  477:     # --- Build command ---
  478:     cmd = [
  479:         claude_cmd, "-p", prompt,
  480:         "--allowedTools", allowed_tools,
  481:         "--max-turns", str(max_turns),
  482:         "--model", model,
  483:         "--output-format", "text",
  484:     ]
  485: 
  486:     # --- Execute ---
  487:     t0 = time.time()
  488:     exit_code = -1
  489:     status = "error"
  490: 
  491:     try:
  492:         # A-IN-3: serialize all in-process Claude invocations so the
  493:         # 8-worker ticker pool / metals fast-tick / signal subprocesses
  494:         # don't spawn 5 concurrent expensive Claude processes.
  495:         # A-IN-2: tree-killing helper for grandchild cleanup on timeout.
  496:         with _invoke_lock:
  497:             rc, _stdout, _stderr, timed_out = _run_with_tree_kill(
  498:                 cmd,
  499:                 timeout=timeout,
  500:                 env=_clean_env(),
  501:                 cwd=working_dir,
  502:                 label=f"claude({caller})",
  503:             )
  504:         if timed_out:
  505:             status = "timeout"
  506:         else:
  507:             exit_code = rc
  508:             status = "invoked" if exit_code == 0 else "error"
  509:             # 2026-04-13: Silent-failure detector. claude CLI can exit 0 while
  510:             # printing "Not logged in" when OAuth/keychain auth can't be read
  511:             # (e.g. --bare flag, missing ANTHROPIC_API_KEY). Override status
  512:             # so the failure surfaces instead of being lost to exit_code=0.
  513:             # BUG-ECHO follow-up (Codex P2 finding 2026-04-16): scan stdout
  514:             # and stderr SEPARATELY rather than concatenating without a
  515:             # newline. Concat-without-newline could merge the marker into
  516:             # the last stdout line ("...stdoutNot logged in"), defeating
  517:             # the start-of-line check shipped today. Scanning each stream
  518:             # independently preserves both streams' line-1 position.
  519:             stdout_hit = detect_auth_failure(
  520:                 _stdout or "", caller,

codex
I’ve resolved the factual status of each cited claim. The remaining work is compressing that into the verdict matrix and calling out one additional orchestration bug that surfaced while tracing the false-return paths.
codex
## Verdicts

- [P0] Timeout watchdog treats `_agent_timeout == 0` as “no timeout” — `portfolio/agent_invocation.py:1142`
  Verdict: FALSE-POSITIVE
  Reason: `_agent_timeout` is only assigned from `TIER_CONFIG`’s hard-coded positive values (`120/600/900`) at `portfolio/agent_invocation.py:174-177`, `:562-563`, and `:867`, so Claude’s `0` scenario is unreachable in this code.

- [P0] Auth-scan byte offset is captured too early and can miss early subprocess output — `portfolio/agent_invocation.py:839`
  Verdict: FALSE-POSITIVE
  Reason: the offset is captured at `:839`, the log is opened at `:840`, and the subprocess is not spawned until `:856`, so there is no window where this agent can emit output before the captured offset.

- [P0] Ranging dampening consumes the consensus baseline even when it suppresses the trigger — `portfolio/trigger.py:231`
  Verdict: PARTIAL
  Reason: in the suppressed branch the code still does `triggered_consensus[ticker] = action` at `portfolio/trigger.py:238-239`, which means a later same-direction higher-confidence signal is no longer “new.”
  Adjustment: the bug is real, but `P0` is inflated; this is a missed-trigger logic bug, not a catastrophic failure.

- [P1] Module-global `_agent_timeout` is stale across reentrant invocations — `portfolio/agent_invocation.py:580`
  Verdict: FALSE-POSITIVE
  Reason: the entire running-agent check is serialized by `_completion_lock` at `portfolio/agent_invocation.py:574`, and if a process is still alive the function returns before any new timeout is assigned (`:575-598`); `_agent_timeout` is only overwritten after a fresh `Popen` succeeds at `:856-867`.

- [P1] `_agent_log_start_offset` is not cleared on completion — `portfolio/agent_invocation.py:1323`
  Verdict: FALSE-POSITIVE
  Reason: every new invocation overwrites `_agent_log_start_offset` before spawn at `portfolio/agent_invocation.py:838-840`, and nothing consults the old value after cleanup unless a new invocation is already setting a fresh one.

- [P1] Stack-overflow auto-disable is permanent once tripped — `portfolio/agent_invocation.py:544`
  Verdict: CONFIRMED
  Reason: `_consecutive_stack_overflows` is persisted at `portfolio/agent_invocation.py:153-171`, `invoke_agent()` hard-skips when it reaches `_MAX_STACK_OVERFLOWS` at `:544-550`, and the only reset path is a later non-stack-overflow completion at `:1308-1313`, which cannot happen once spawns are blocked.

- [P1] Startup grace is ignored on in-process loop restart — `portfolio/trigger.py:189`
  Verdict: FALSE-POSITIVE
  Reason: the real restart mechanism here is process restart, and `check_triggers()` already keys grace to PID change at `portfolio/trigger.py:168-177`; Claude’s “restart without reimport” path is hypothetical and not implemented anywhere in this subsystem.

- [P1] Decision-feedback does a full journal scan on every invocation — `portfolio/agent_invocation.py:293`
  Verdict: PARTIAL
  Reason: `_build_decision_feedback()` calls `load_jsonl(JOURNAL_FILE)` with no limit at `portfolio/agent_invocation.py:293`, and `portfolio/file_utils.py:104-133` reads the whole JSONL file before returning, so the later `max_entries` break only saves post-processing.
  Adjustment: the performance concern is real, but `P1` overstates it absent evidence that current journal size is already causing user-visible latency.

- [P1] `ThreadPoolExecutor.cancel_futures` does not stop running ticker workers — `portfolio/main.py:658`
  Verdict: PARTIAL
  Reason: `f.cancel()` plus `pool.shutdown(wait=False, cancel_futures=True)` at `portfolio/main.py:657-661` cannot stop already-running tasks, and the code even documents “5 zombie threads” finishing later at `:580-583`.
  Adjustment: this is a known Python-thread limitation and explicit tradeoff, not a hidden “hard-killable” leak at line 658.

- [P1] Timeout kill failures are not persisted distinctly — `portfolio/agent_invocation.py:1148`
  Verdict: PARTIAL
  Reason: `_check_agent_completion_locked()` ignores the boolean returned by `_kill_overrun_agent()` at `portfolio/agent_invocation.py:1145`, while `_kill_overrun_agent()` always logs plain `status="timeout"` via `_log_trigger()` at `:528-533` / `:243-251` and clears `_agent_proc` at `:535` even when `kill_ok` is false.
  Adjustment: the line citation is wrong and the impact is understated; the real bug is not just missing `kill_status`, but that a failed kill can be forgotten entirely.

- [P1] `classify_tier()` uses UTC weekday for market-hours decisions — `portfolio/trigger.py:434`
  Verdict: FALSE-POSITIVE
  Reason: `market_open` is gated by UTC hours from `_eu_market_open_hour_utc()` / `_market_close_hour_utc()` at `portfolio/trigger.py:449-453` and `portfolio/market_timing.py:53-100`; the UTC/local weekday mismatch Claude cites only occurs near midnight, outside the 07/08-20/21 UTC market-hours window.

- [P1] Heartbeat keepalive daemon can corrupt `health_state.json` on shutdown — `portfolio/health.py:140`
  Verdict: FALSE-POSITIVE
  Reason: `heartbeat()` writes under `_health_lock` using `atomic_write_json()` at `portfolio/health.py:83-86`, and `portfolio/file_utils.py:45-63` explicitly uses tempfile + `os.replace`, so the failure mode is a missed final beat, not a torn JSON file.

- [P1] Broad `except Exception` hides `ImportError` and falls through to `.bat` invocation — `portfolio/agent_invocation.py:809`
  Verdict: FALSE-POSITIVE
  Reason: the cited line is just `claude_cmd = shutil.which("claude")` at `portfolio/agent_invocation.py:809`; there is no import there, and the broad `except` at `:932-936` logs the failure and returns `False` rather than silently falling back to the batch file.

## New findings (you, not Claude)

- [P2] `main.run()` mislabels every `invoke_agent(False)` outcome as `skipped_busy`, producing duplicate or wrong invocation rows — `portfolio/main.py:849`
  `main.run()` does `_log_trigger(reasons_list, "invoked" if result else "skipped_busy", tier=tier)` at `portfolio/main.py:846-849`, but `invoke_agent()` returns `False` for many non-busy cases: `skipped_stack_overflow` (`portfolio/agent_invocation.py:544-550`), `skipped_gate` (`:616-619`), drawdown blocks (`:646-648`, `:660-661`), trade-guard blocks (`:735-742`), and generic spawn failures (`:932-936`). Several of those paths already write a precise status themselves, so the caller appends a second contradictory `skipped_busy` row; others get mislabeled as “busy” even when the real problem was an error or policy block.

## Summary
- Confirmed: 1
- Partial: 4
- False-positive: 8
- New from you: 1
tokens used
177,980
## Verdicts

- [P0] Timeout watchdog treats `_agent_timeout == 0` as “no timeout” — `portfolio/agent_invocation.py:1142`
  Verdict: FALSE-POSITIVE
  Reason: `_agent_timeout` is only assigned from `TIER_CONFIG`’s hard-coded positive values (`120/600/900`) at `portfolio/agent_invocation.py:174-177`, `:562-563`, and `:867`, so Claude’s `0` scenario is unreachable in this code.

- [P0] Auth-scan byte offset is captured too early and can miss early subprocess output — `portfolio/agent_invocation.py:839`
  Verdict: FALSE-POSITIVE
  Reason: the offset is captured at `:839`, the log is opened at `:840`, and the subprocess is not spawned until `:856`, so there is no window where this agent can emit output before the captured offset.

- [P0] Ranging dampening consumes the consensus baseline even when it suppresses the trigger — `portfolio/trigger.py:231`
  Verdict: PARTIAL
  Reason: in the suppressed branch the code still does `triggered_consensus[ticker] = action` at `portfolio/trigger.py:238-239`, which means a later same-direction higher-confidence signal is no longer “new.”
  Adjustment: the bug is real, but `P0` is inflated; this is a missed-trigger logic bug, not a catastrophic failure.

- [P1] Module-global `_agent_timeout` is stale across reentrant invocations — `portfolio/agent_invocation.py:580`
  Verdict: FALSE-POSITIVE
  Reason: the entire running-agent check is serialized by `_completion_lock` at `portfolio/agent_invocation.py:574`, and if a process is still alive the function returns before any new timeout is assigned (`:575-598`); `_agent_timeout` is only overwritten after a fresh `Popen` succeeds at `:856-867`.

- [P1] `_agent_log_start_offset` is not cleared on completion — `portfolio/agent_invocation.py:1323`
  Verdict: FALSE-POSITIVE
  Reason: every new invocation overwrites `_agent_log_start_offset` before spawn at `portfolio/agent_invocation.py:838-840`, and nothing consults the old value after cleanup unless a new invocation is already setting a fresh one.

- [P1] Stack-overflow auto-disable is permanent once tripped — `portfolio/agent_invocation.py:544`
  Verdict: CONFIRMED
  Reason: `_consecutive_stack_overflows` is persisted at `portfolio/agent_invocation.py:153-171`, `invoke_agent()` hard-skips when it reaches `_MAX_STACK_OVERFLOWS` at `:544-550`, and the only reset path is a later non-stack-overflow completion at `:1308-1313`, which cannot happen once spawns are blocked.

- [P1] Startup grace is ignored on in-process loop restart — `portfolio/trigger.py:189`
  Verdict: FALSE-POSITIVE
  Reason: the real restart mechanism here is process restart, and `check_triggers()` already keys grace to PID change at `portfolio/trigger.py:168-177`; Claude’s “restart without reimport” path is hypothetical and not implemented anywhere in this subsystem.

- [P1] Decision-feedback does a full journal scan on every invocation — `portfolio/agent_invocation.py:293`
  Verdict: PARTIAL
  Reason: `_build_decision_feedback()` calls `load_jsonl(JOURNAL_FILE)` with no limit at `portfolio/agent_invocation.py:293`, and `portfolio/file_utils.py:104-133` reads the whole JSONL file before returning, so the later `max_entries` break only saves post-processing.
  Adjustment: the performance concern is real, but `P1` overstates it absent evidence that current journal size is already causing user-visible latency.

- [P1] `ThreadPoolExecutor.cancel_futures` does not stop running ticker workers — `portfolio/main.py:658`
  Verdict: PARTIAL
  Reason: `f.cancel()` plus `pool.shutdown(wait=False, cancel_futures=True)` at `portfolio/main.py:657-661` cannot stop already-running tasks, and the code even documents “5 zombie threads” finishing later at `:580-583`.
  Adjustment: this is a known Python-thread limitation and explicit tradeoff, not a hidden “hard-killable” leak at line 658.

- [P1] Timeout kill failures are not persisted distinctly — `portfolio/agent_invocation.py:1148`
  Verdict: PARTIAL
  Reason: `_check_agent_completion_locked()` ignores the boolean returned by `_kill_overrun_agent()` at `portfolio/agent_invocation.py:1145`, while `_kill_overrun_agent()` always logs plain `status="timeout"` via `_log_trigger()` at `:528-533` / `:243-251` and clears `_agent_proc` at `:535` even when `kill_ok` is false.
  Adjustment: the line citation is wrong and the impact is understated; the real bug is not just missing `kill_status`, but that a failed kill can be forgotten entirely.

- [P1] `classify_tier()` uses UTC weekday for market-hours decisions — `portfolio/trigger.py:434`
  Verdict: FALSE-POSITIVE
  Reason: `market_open` is gated by UTC hours from `_eu_market_open_hour_utc()` / `_market_close_hour_utc()` at `portfolio/trigger.py:449-453` and `portfolio/market_timing.py:53-100`; the UTC/local weekday mismatch Claude cites only occurs near midnight, outside the 07/08-20/21 UTC market-hours window.

- [P1] Heartbeat keepalive daemon can corrupt `health_state.json` on shutdown — `portfolio/health.py:140`
  Verdict: FALSE-POSITIVE
  Reason: `heartbeat()` writes under `_health_lock` using `atomic_write_json()` at `portfolio/health.py:83-86`, and `portfolio/file_utils.py:45-63` explicitly uses tempfile + `os.replace`, so the failure mode is a missed final beat, not a torn JSON file.

- [P1] Broad `except Exception` hides `ImportError` and falls through to `.bat` invocation — `portfolio/agent_invocation.py:809`
  Verdict: FALSE-POSITIVE
  Reason: the cited line is just `claude_cmd = shutil.which("claude")` at `portfolio/agent_invocation.py:809`; there is no import there, and the broad `except` at `:932-936` logs the failure and returns `False` rather than silently falling back to the batch file.

## New findings (you, not Claude)

- [P2] `main.run()` mislabels every `invoke_agent(False)` outcome as `skipped_busy`, producing duplicate or wrong invocation rows — `portfolio/main.py:849`
  `main.run()` does `_log_trigger(reasons_list, "invoked" if result else "skipped_busy", tier=tier)` at `portfolio/main.py:846-849`, but `invoke_agent()` returns `False` for many non-busy cases: `skipped_stack_overflow` (`portfolio/agent_invocation.py:544-550`), `skipped_gate` (`:616-619`), drawdown blocks (`:646-648`, `:660-661`), trade-guard blocks (`:735-742`), and generic spawn failures (`:932-936`). Several of those paths already write a precise status themselves, so the caller appends a second contradictory `skipped_busy` row; others get mislabeled as “busy” even when the real problem was an error or policy block.

## Summary
- Confirmed: 1
- Partial: 4
- False-positive: 8
- New from you: 1
