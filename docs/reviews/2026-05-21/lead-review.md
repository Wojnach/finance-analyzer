# Lead Adversarial Review — 2026-05-21

Baseline SHA: `604f0ef1` (HEAD on main after PLAN commit; reviewed code at b39f5b3e contents).
Reviewer: lead, independent fresh-read.
Goal: surface real-money / silent-failure / data-corruption risks. Skip style.

Format: `path:line: <severity>: <problem>. Fix: <suggestion>.`

## P0 / P1 — real-money or silent-failure path

portfolio/exit_optimizer.py:374: P1: knockout distance calc `(market.price - position.financing_level) / market.price * 100` assumes LONG MINI (financing < price). For Bear MINI (financing > price) distance is negative; `if distance_pct < 3` always trips → forces market exit on every Bear MINI evaluation. Fix: branch on `position.financing_level > market.price` for Bear; use `abs(market.price - financing_level)/market.price` and direction-aware sign.

portfolio/exit_optimizer.py:431-451: P1: same Bear-MINI long-bias in `_apply_risk_overrides` (knockout proximity + p_knockout using `session_min`). For Bear MINIs the relevant breach is `session_max >= financing_level`, not `session_min <= stop_buffer`. Fix: direction-aware sentinel chosen from financing-vs-price relationship.

portfolio/risk_management.py:374: P1: `compute_stop_levels` produces ATR-based stop without consulting `financing_level`. ATR is capped at 15% so stop floor = entry × 0.70. For a MINI with knockout 5% below entry, stop sits below barrier → instant-fill before stop ever fires. Fix: when ticker is a warrant, max(atr_stop, financing_level * 1.03) and surface the clamp.

portfolio/agent_invocation.py:1018: P1: Layer 2 `--allowedTools Edit,Read,Bash,Write` lets the subprocess write/edit ANY file in the working tree (including portfolio_state.json and the config.json symlink). Bypasses atomic_write_json semantics; one bad agent run can corrupt state. Fix: pin allowedTools to Read,Bash only — let Bash invoke the journal/telegram helpers; remove direct Edit/Write.

portfolio/agent_invocation.py:1056: P1: `PF_HEADLESS_AGENT=1` documents "log unresolved critical errors and proceed". Current journal shows 7 contract_violations TODAY (Layer 2 silent fails for XAU/XAG/ETH/BTC triggers) — the headless agent will dutifully log them and proceed, never recovering. No self-healing path. Fix: when a triggering category matches the headless context (e.g., the same ticker that just contract-violated), gate to T3 escalation or quarantine the trigger.

portfolio/file_utils.py:316-354: P1: `last_jsonl_entry` reads only the last 4 KB. Layer 2 journal entries can routinely exceed 4 KB (multi-paragraph reasoning + thesis blocks). When entry > 4 KB, the parser sees a mid-line head and skips it, returning the PRIOR entry or None. agent_invocation.py:1468 uses this to extract `fishing_context` from "the new journal entry" — silent fishing-context-from-stale-entry. Fix: grow the tail buffer dynamically (mirror `_read_tail_with_growth` in dashboard/app.py:134), or scan from the trailing newline backwards.

portfolio/portfolio_mgr.py:90: P1: portfolio-state corruption only emits `logger.critical(...)` — does not append to `data/critical_errors.jsonl`. The critical-errors journal is what drives the auto-fix dispatcher and surfaces problems at session start. A genuinely corrupt portfolio file (e.g., antivirus mid-write quarantine) gets recovered from backup silently with no oncall signal. Fix: also append a `category=portfolio_corruption` entry to critical_errors.jsonl when recovery happens.

portfolio/trigger.py:373-400: P1: sustained-flip duration gate (`SUSTAINED_DURATION_S = 120`) plus count gate (`SUSTAINED_CHECKS = 3`) are OR'd at line 373. At a 60s cadence this means a SINGLE bar flip that stays 120s wall-clock fires as "sustained" — count=2 (only two cycles in 120s). Documentation says "sustained" implies persistence across N consecutive cycles. The OR weakens the guarantee silently. Fix: AND the two gates, or document the floor as min(count, duration) explicitly.

dashboard/app.py:78-95: P1: TTL cache backs every route. Cache shared globally across threads (Flask threaded server). Sensitive endpoints (`/api/portfolio`, `/api/grid-fisher`, `/api/decisions`) cache for 5s. If a user requests a route during a write window and another user reads the same route within 5s, both see the stale snapshot. Acceptable for read-only dashboard; check that no MUTATING route relies on `_cached_read`. Audit needed.

portfolio/agent_invocation.py:949-967: P1: multi-agent mode runs `wait_for_specialists(procs, timeout=30)` synchronously inside `invoke_agent`, blocking the main loop for up to 30s before the parent Claude subprocess even spawns. Production main loop cycle target is 60s; a multi-agent T2 burns half the cycle just waiting. The TODO comment acknowledges this. Under sustained triggers the loop will fall behind cadence. Fix: launch specialists in a background thread, gather results when the parent agent completes (or skip multi-agent for T2 and reserve for T3 only).

portfolio/avanza_session.py:720-811: P1 (defense-in-depth): `place_stop_loss` does not validate `trigger_price` against the warrant barrier. A caller with a buggy financing_level lookup could place a stop AT or below the barrier — the stop never fires before instant knockout. There is no barrier-awareness layer between the caller and the API. Fix: when account_id-by-orderbook resolves to a MINI futures warrant, fetch the financing_level (`get_warrant_details`) and refuse `trigger_price <= financing_level * 1.03`.

portfolio/trigger.py:278-280: P1: post-trade reassessment trigger fires on EVERY new transaction without cooldown. Combined with Layer 2 agents that submit BUY then SELL (or laddered orders), a single agent run can fire 2-3 post-trade triggers, each producing another Layer 2 spawn within seconds. The flip cooldown (30 min) does not cover this path. Fix: add `POST_TRADE_COOLDOWN_S=300` keyed off `last_post_trade_trigger_time` in trigger_state.

portfolio/agent_invocation.py:1503-1518: P1: stub journal entry on `incomplete` writes `decisions: {patient: HOLD, bold: HOLD}`. Analytics consume `journal.decisions[strategy].action` as truth. A T1 quick-check that returned incomplete is now logged as if Layer 2 explicitly said HOLD on both strategies → accuracy stats for "Layer 2 HOLD" inflate, and the agent's true decision distribution is poisoned. Fix: use `action="NO_DECISION"` (or omit `decisions` entirely) so downstream code can distinguish "agent chose HOLD" from "agent crashed".

portfolio/agent_invocation.py:704-706: P1: `recent = load_jsonl(INVOCATIONS_FILE)` loads the FULL invocations file into memory every invocation, then takes `[-50:]`. With months of triggers this file grows steadily; the helper exists already (`load_jsonl_tail`). Fix: use `load_jsonl_tail(INVOCATIONS_FILE, max_entries=50)`.

## P2 — should-fix soon

portfolio/exit_optimizer.py:556-598: P2: candidate generation iterates `quantiles` of `session_max` only. Long-only design. SHORT positions (none today, but Bear-MINI universe is plausible per Avanza catalog) need quantiles of `session_min`. Fix: bifurcate by `position.side` (default `LONG`), select source array accordingly.

portfolio/exit_optimizer.py:177-185: P2: `_estimate_volatility` falls back to a fixed 20% annualized when market.volatility_annual and market.atr_pct are both missing. For a quiet day this dramatically over-states risk; for a stressed day it under-states. Better signal: scale from instrument-typical ATR for that ticker (use accuracy_stats observed ATR). Fix: per-ticker priors keyed off recent ATR p50.

portfolio/agent_invocation.py:1429-1432: P2: `status="success"` requires BOTH `journal_written` AND `telegram_sent`. A network blip during Telegram send sets `status="incomplete"`, sends a `*L2 INCOMPLETE*` alert (line 1497), AND triggers the false-positive stub-entry path. Telegram failures are noisy and not safety-critical. Fix: split — `journal_written` is the success bar; Telegram failure logs a softer notice.

portfolio/agent_invocation.py:1520-1544: P2: stack-overflow auto-disable after 5 consecutive crashes. Once tripped, only manual reset clears it (counter persisted). If a transient Claude CLI bug stabilizes after restart, system stays disabled forever. Fix: auto-decay one count per ` (consecutive >0 AND no SO in last 24h)`.

portfolio/agent_invocation.py:611-678: P2: `_kill_overrun_agent` unconditionally sets `_agent_proc = None` even when kill_ok=False. Caller barrier in `invoke_agent:768` exists, but other callers that check `_agent_proc is None` to detect "is work running" will see "no work" while the OS process is still alive. Fix: only null `_agent_proc` when kill confirmed.

portfolio/trigger.py:191-262: P2: `_save_state` mutates the dict (pruning `_current_tickers`) and atomically writes. Two concurrent callers would race the prune logic AND the write (last-write-wins). Today only main.py calls it serially. Document the single-writer assumption explicitly. Fix: `@requires(single_writer)` comment, or guard with a process-level file lock.

portfolio/trigger.py:165-169: P2: `_today_str` uses UTC. Trade decisions and accuracy windows are user-CET. A trigger fired at 23:55 UTC (00:55 CET next day) increments `last_trigger_date` on the UTC day, not the CET day → "first of day" Tier-3 promotion happens at 01:00 CET instead of 00:00 CET. Fix: use Europe/Stockholm zone for the date string.

portfolio/risk_management.py:308: P2: drawdown circuit breaker compares `current_drawdown_pct > max_drawdown_pct`. But agent_invocation.py:834 calls `check_drawdown(..., max_drawdown_pct=_DRAWDOWN_WARN_PCT=20)` — this means the *advisory* threshold runs the breach math; the BLOCK threshold (50) is only compared against the returned `current_drawdown_pct`. The `breached` bool in the dict is therefore effectively unused, and a caller that trusted `dd["breached"]` would block at 20% even though policy is 50%. Fix: pass the BLOCK threshold to check_drawdown and trust `breached`, OR keep advisory shape but rename to `advisory_breached`.

portfolio/file_utils.py:144-207: P2: `load_jsonl_tail` default `tail_bytes=512_000`. For a JSONL with rare-but-huge entries (Layer 2 journal sometimes >4 KB), tail of 512 KB ~= 100-1000 entries depending on size variance. Caller asking `max_entries=500` may under-deliver silently. Fix: copy the `_read_tail_with_growth` pattern from dashboard into file_utils itself.

portfolio/portfolio_mgr.py:35: P2: `_state_locks` dict grows per-unique-path with no eviction. Tests can leak unique paths via tmp_path. Documented as 3 paths in prod, but if any future test imports state-management with non-tmp paths, this dict grows. Fix: WeakValueDictionary keyed on path string, or bounded LRU.

portfolio/portfolio_mgr.py:111-114: P2: `_save_state_to` rotates backups (shutil.copy2 → not atomic) INSIDE the lock, then atomic_write_json. Reader between the two writes sees a state that matches `.bak` but no longer matches `path` — fine. But if `_rotate_backups` partially completes (disk-full mid-copy) and the lock is released, the next save sees a torn backup chain. Backup-chain integrity check missing. Fix: try/except around _rotate_backups; on failure, skip backup and proceed (still atomic-write the primary file).

portfolio/signal_engine.py:1014-1020: P2: `MIN_VOTERS_METALS = 2` deviates from `.claude/rules/signals.md` which declares "MIN_VOTERS = 3 for all asset classes". Documented as a 2026-05-11 fix for noisier intraday metals horizon. Inconsistency between source code and rules-of-record. Fix: update `.claude/rules/signals.md` to mention the metals=2 exception, or revert to 3 with persistence filter compensation.

portfolio/trigger.py:373-388: P2: flip cooldown applies only to "section 2" sustained flips. Section 1 (consensus-from-HOLD) has no cooldown — a ticker that oscillates HOLD↔BUY↔HOLD↔BUY (e.g., MSTR overnight) can fire consensus triggers indefinitely. Fix: extend `FLIP_COOLDOWN_S` to section 1.

portfolio/trigger.py:330-336: P2: when consensus drops from BUY/SELL → HOLD, baseline reset to HOLD silently (no Layer 2 trigger). But "consensus cleared" is itself a tradable event — closing a position when conviction evaporates. Documented as design choice (sustained flip handles direction flips). Acceptable but check that exit-management logic catches the implicit consensus-loss. P3.

portfolio/agent_invocation.py:1037-1038: P2: `_agent_log_start_offset` is taken BEFORE opening the log fh in append mode; if another writer appends between the stat() and open(), the offset under-reads. Race window small. Fix: re-stat AFTER open().

dashboard/app.py:80-95: P2: `_cache` shared across all routes with no per-key isolation of expensive vs cheap reads. A slow endpoint that grabs `_cache_lock` during read blocks all other dashboard requests. Fix: per-key locks or asyncio-friendly pattern.

dashboard/app.py:44-49: P2: `_ALLOWED_ORIGINS` whitelists localhost + 127.0.0.1. The dashboard binds dual-stack IPv4+IPv6 (per CLAUDE.md) and accepts Cloudflare Access header bypass. CORS check is origin-only; if an attacker can spoof the Origin header (they can — browsers send it but tools don't validate), the Cloudflare path can be impersonated. Fix: verify CF-Access JWT signature, not just header presence.

portfolio/agent_invocation.py:1043-1048: P2: NODE_OPTIONS appended via string concat with existing value. Existing value might end with `--stack-size=8192`; the new option `--stack-size=16384` doesn't override — Node uses the LAST occurrence, so it works, but the env grows on each invocation unless main.py is a fresh process. Production main.py is long-lived → env stays bounded. P3.

portfolio/file_utils.py:74-94: P2: `load_json` returns the `default` on JSONDecodeError. For the most critical files (portfolio_state.json, accuracy_cache.json), corruption is masked as "missing". Recommend `require_json` (already exists) for these. Audit callers. Fix: list of `STRICT_JSON_PATHS` enforced via wrapper.

portfolio/trigger.py:621-624: P2: off-hours periodic review caps at T1. Per the comment "save T3 budget for market hours". But a sustained off-hours move on crypto (24/7) would never escalate to T3 — even a real breakout doesn't get full analysis. Fix: allow T3 for triggers that include crypto tickers off-hours.

portfolio/risk_management.py:373: P2: ATR capped at 15.0 before 2x multiplication. Stop floor = entry × 0.70 = 30% loss tolerance. Bold strategy budget docs target 10-20% knockout risk per position. 30% is over the policy. Cap probably tuned for warrant whipsaws; fine in context but should annotate that this overrides the 10-20% policy.

portfolio/exit_optimizer.py:392-393: P2: `HOLD_TIME_EXTENDED` triggered at 5h hold. User memory says metals bounce trades are 3-5h max. Threshold matches policy but only as advisory flag, not as override. Fix: promote to risk override when hold > policy max.

portfolio/portfolio_mgr.py:65-75: P2: `_validated_state` merges loaded into defaults. If a NEW required field is added later, OLD loaded states won't have it; the default will fill it but loaded state still satisfies `isinstance(loaded, dict)` so we never log a migration. Fix: explicit schema version + migration table.

## P3 — nit-but-noteworthy

portfolio/agent_invocation.py:268-307: P3: ticker extraction via regex `\b([A-Z]{2,5}-USD)\b` then fallback `\b([A-Z]{2,5})\b` with lookahead. Fails on tickers like `BRK.B` or `XAG`-suffix variants. Default `XAG-USD` masks the failure. Fine for current universe; document the limitation.

portfolio/agent_invocation.py:1018: P3: model pinned to `claude-sonnet-4-6`. When 4.7 stabilizes for the user, this needs manual update. Fix: pull from config.layer2.model with sensible default.

portfolio/trigger.py:114: P3: `FLIP_COOLDOWN_S = 1800` (30 min) per ticker — interacts with the rules-doc "30 min flip cooldown". OK; could be config-driven for tuning.

portfolio/trigger.py:121-122: P3: `RANGING_CONSENSUS_MIN_CONFIDENCE = 0.40`. Magic number; pull from config.

dashboard/app.py:210-221: P3: `_hours_until_stockholm_close` defaults to 21:30 close, but Avanza commodity warrants close 21:55. Inconsistency between dashboard and trading flow.

dashboard/app.py:317-319: P3: `state["theta_in"]` etc default to magic constants if not in config. OK; document.

## Patterns

- Long-bias assumption is pervasive in exit/risk modules (exit_optimizer, risk_management). Adding Bear MINIs to the universe will trigger multiple correlated bugs.
- Silent default-on-failure (`load_json(..., default={})`) used in safety-critical paths (portfolio state, accuracy cache, agent_summary). The `require_json` strict helper exists but is under-used.
- "Status incomplete" stub-entry path turns L2 failures into HOLD decisions in analytics, polluting the very accuracy stats Layer 2 needs to learn from.
- Trigger system has multiple SKIP paths (perception_gate, no_position_skip, drawdown block, trade_guards block, auth cooldown) but the loop_contract that audits "Layer 2 trigger fired but no journal" doesn't necessarily know about all of them — explains 7 contract_violations TODAY.
- Subprocess/external-call timeouts (Avanza, Claude CLI, Telegram) are individually retried but no shared backoff registry — concurrent failures stack.

## Findings the lead is most worried about

1. exit_optimizer/risk_management Bear-MINI long bias — only matters when SHORT MINIs are traded; ship a guard before that universe expansion.
2. agent_invocation tools=Edit,Write — single Claude misbehavior corrupts state outside atomic_write_json.
3. last_jsonl_entry 4 KB tail truncating fishing-context extraction silently.
4. trigger contract violations TODAY (10 unresolved critical errors): root cause likely the gap between "trigger fires" and "invoke_agent returns False due to a gate" with no journal entry — loop_contract sees the gap and alarms. Need to thread the SKIP outcomes back to loop_contract.

## Out of Scope but Spotted

- `data/metals_loop.py` is 7880 LoC — a single-file metals subsystem. Modularity hazard; refactor backlog should treat this as a Tier-1 item.
- `data/critical_errors.jsonl` has 10 unresolved entries from the past 7d that the dispatcher should have addressed automatically. PF-FixAgentDispatcher is per-category cooldowned; verify the cooldown didn't disable a recurring category.
- Layer 2 silent failures TODAY (2026-05-21) per critical_errors.jsonl: BTC, XAU, XAG, ETH triggers each had 13-20min gaps between trigger fire and journal write. If true silent failures, the watchdog should be killing them; if the gates are skipping legitimately, the loop_contract logic needs an update.
