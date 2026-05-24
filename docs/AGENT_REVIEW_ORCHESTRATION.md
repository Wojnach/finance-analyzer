# Orchestration Subsystem — Adversarial Code Review (2026-05-24)

**Scope:** `portfolio/main.py`, `portfolio/agent_invocation.py`, `portfolio/autonomous.py`, `portfolio/trigger.py`, `portfolio/trigger_buffer.py`, `portfolio/multi_agent_layer2.py`, `portfolio/escalation_gate.py`, `portfolio/escalation_router.py`, `portfolio/market_timing.py`, `portfolio/reporting.py`, `portfolio/perception_gate.py`, `portfolio/loop_contract.py`, `portfolio/loop_health.py`, `portfolio/loop_processes.py`, `portfolio/macro_context.py`, `portfolio/regime_alerts.py`, `portfolio/digest.py`, `portfolio/daily_digest.py`, `portfolio/weekly_digest.py`, `portfolio/session_calendar.py`

**Worktree:** `Q:\finance-analyzer\finance-analyzer-reviews\2026-05-24` (branch `review/fgl-2026-05-24`)

**Method:** Whole-codebase audit. Treated all files as fresh additions; ignored prior P0/P1 closure markers unless verified in code.

---

## TOP 5 MUST-FIX

1. **`portfolio/main.py:972-979` | P0 | layer2/off-hours-silent-skip** — Layer 2 silently skipped when `_is_agent_window()` returns False, with NO autonomous fallback. `_is_agent_window` is weekday + US-business-hours bound, but XAU/XAG/BTC/ETH trade 24/7. Weekend XAG trigger → `"skipped_offhours"` logged, no journal, no Telegram, no decision. Prior 2026-05-19 P0 was fixed only in the autonomous-first branch (947-959); this bare `elif layer2_cfg.get("enabled", True):` branch was missed. Fix: invoke `autonomous_decision(...)` in the off-hours `else`.

2. **`portfolio/agent_invocation.py:402-423` | P0 | layer2/broken-skip-gate** — `_no_position_skip()` reads `signals` key from `agent_context_t1.json`, but `_write_tier1_summary()` (`reporting.py:1191-1247`) writes only `held_positions`, `all_prices`, `macro_headline` — NO `signals` field. When `claude_budget.no_position_skip_enabled=True` and no positions are held, the gate ALWAYS returns `(True, "no_position_no_entry")` regardless of signal strength → every Layer 2 trigger silently skipped. Latent (off by default) but a one-flag flip breaks Layer 2 entirely. Fix: read `weighted_confidence` from `agent_summary_compact.json`, or populate `signals.<ticker>.weighted_confidence` in `_write_tier1_summary`.

3. **`portfolio/agent_invocation.py:1021` | P0 | multi-agent/specialist-timeout** — Default `specialist_timeout_s=30`. Each specialist runs `max_turns=8-10` × ~10-15s/turn ≈ 80-120s on Sonnet. With 30s budget, `success_count==0` is the norm → trips `specialist_quorum_fail` at line 1025-1046 → `return` (None) → main.py:951 logs duplicate `skipped_busy_<why>`. Multi-agent mode effectively broken with default config. Fix: bump default to ≥150s (max specialist intrinsic timeout + buffer), or align with `SPECIALISTS[*].timeout`.

4. **`portfolio/main.py:1085-1091` and `:1050` | P0 | cycle-id-restart-aliasing** — IC cache refresh (`% 60 == 30`) and safeguard checks (`% 100 == 0`) gated on `_run_cycle_id`, an in-memory counter reset on every process restart. Comment claims "every 60 cycles ≈ 60 min" but cycles are 600s since 2026-04-09 → real cadence is 10h IF the process stays up. Frequent restarts (Task Scheduler `auto-restart 30s`) can make these branches fire every cycle (restart-loop) or never (restart pattern unfavourable). IC cache silently goes stale; safeguard checks (outcome staleness, dead signals) skip indefinitely. Fix: gate on wall-clock ts stored in shared_state, like `_last_log_rotation_ts` at line 404 already does.

5. **`portfolio/loop_contract.py:335` | P0 | autonomous-first/false-positive** — In-flight suppression matches `status == "invoked"` exactly, but autonomous-first path (`main.py:951`) writes `f"invoked_{why}"` like `"invoked_drawdown_+5.5pct"`. Once `autonomous_first_enabled=true`, every Claude escalation gets a non-matching status → in-flight check fails → `layer2_journal_activity` fires false-positive violations every cycle once grace elapses → critical_errors.jsonl floods → fix-agent dispatcher wakes up against nothing. Fix: `status == "invoked" or status.startswith("invoked_")`.

---

## ALL FINDINGS

Format: `path:line | severity | category | description | fix`

### Critical (P0, 90-100)

`portfolio/main.py:972-979` | P0 | layer2/off-hours-silent-skip | Off-hours branch returns with no fallback. 24/7 instruments lose all decision coverage on weekends and overnight EU/US-closed. The autonomous-first branch above DOES fall back; this branch missed the update. | Fix: invoke `autonomous_decision(...)` here under `heartbeat_keepalive`.

`portfolio/agent_invocation.py:402-423` | P0 | layer2/broken-skip-gate | `_no_position_skip` reads `signals` field from `agent_context_t1.json` which `_write_tier1_summary` doesn't write. Force-skip whenever positions are zero + flag enabled. | Fix: read from `agent_summary_compact.json` OR add `signals` to `_write_tier1_summary` payload.

`portfolio/agent_invocation.py:1018-1046` | P0 | multi-agent/timeout-default | 30s default is below every specialist's intrinsic budget. Quorum-fail every invocation. | Fix: `specialist_timeout_s` default = `max(SPECIALISTS[*].timeout)+30 = 150s`.

`portfolio/main.py:1085-1091` | P0 | cycle-id-restart-aliasing | IC cache + safeguards gated on `_run_cycle_id` (in-memory, restart-reset). Combined with the 2026-04-09 cadence bump to 600s, the actual cadence is "10h IF no restart" → effectively dead. | Fix: monotonic-wallclock gate (`shared_state._last_ic_refresh_ts >= 3600`).

`portfolio/loop_contract.py:335` | P0 | autonomous-first/false-positive-storm | `status == "invoked"` exact-match misses `"invoked_<why>"` from autonomous-first path. In-flight suppression fails → false-positive violation every cycle → critical_errors flood. | Fix: `status == "invoked" or status.startswith("invoked_")`.

### Important (P1, 80-89)

`portfolio/agent_invocation.py:1046` | P1 | multi-agent/early-return-double-log | Bare `return` (None) on specialist_quorum_fail. Caller treats None as falsy → second `_log_trigger` writes `skipped_busy_<why>`. Two rows per trigger. | Fix: `return False`, single canonical log.

`portfolio/multi_agent_layer2.py:198-242` | P1 | wait-for-specialists-sequential | `proc.wait(timeout=remaining)` iterated SEQUENTIALLY; third specialist gets remaining-budget after first two used theirs. Not true parallel wait despite parallel launch. | Fix: `concurrent.futures.wait(FIRST_COMPLETED)` loop.

`portfolio/agent_invocation.py:799-801` | P1 | dead-path-but-no-log | `invoke_agent` returns False on `l2_cfg.get("enabled", True)==False` without `_log_trigger`. Unreachable from main loop (caller routes around it) but breaks any direct external caller (tests, REPL). | Fix: log `_log_trigger(reasons, "skipped_disabled", tier=tier)`.

`portfolio/trigger.py:233` | P1 | set-not-json-serializable | `state["_current_tickers"] = set(...)` is unserializable. Line 195 pops before `atomic_write_json`, but a raise between the assignment and pop crashes the next save. | Fix: store list, or pass through a sentinel kwarg rather than embedded.

`portfolio/trigger.py:336-343` | P1 | partial-state-on-buffer-defer | `triggered_consensus[ticker] = action` writes baseline at line 335 even when the trigger is buffer-deferred (main.py:828 sets `triggered=False`). Buffer eventually flushes → reason claims a transition already in the persisted baseline; next cycle has no re-fire path. | Fix: defer `triggered_consensus` write until after main.py confirms the trigger actually fires.

`portfolio/main.py:807-837` | P1 | trigger-buffer-baseline-skew | When buffer holds back, `state["last"]` baseline (prices, fear_greeds, sentiments) at trigger.py:494-516 is NOT updated. Buffer flushes 5min later → next cycle compares against pre-buffer baseline. Sentiment reversal detection becomes lossy. | Fix: in main.py buffer-defer path, separately call a baseline-update so subsequent cycles compare against current state.

`portfolio/agent_invocation.py:762-783` | P1 | auth-cooldown-bounded-lookback | `recent[-50:]` may push a real auth_error out of view if there's a burst of skipped entries. | Fix: filter by ts cutoff (last 30min) first, then look for auth_error.

`portfolio/autonomous.py:100-101` | P1 | full-jsonl-load-per-cycle | `load_jsonl(JOURNAL_FILE, limit=5)` reads from file start. After thousands of entries this becomes O(N) per cycle. | Fix: `load_jsonl_tail(JOURNAL_FILE, max_entries=5)`.

`portfolio/loop_contract.py:1247-1265` | P1 | journal-uniqueness-readlines | `f.readlines()` over `layer2_journal.jsonl` (up to 5000 entries × few KB after prune) per cycle. Same anti-pattern fixed by BUG-109/190 elsewhere. | Fix: `load_jsonl_tail(LAYER2_JOURNAL_FILE, max_entries=_JOURNAL_UNIQUENESS_TAIL)`.

`portfolio/loop_contract.py:1254` | P1 | journal-jsonl-utf8-strict | `open(...encoding="utf-8")` will raise UnicodeDecodeError on a half-written entry; existing `try/except OSError` doesn't catch it → silent check skip. | Fix: `errors="replace"` or `except (OSError, UnicodeDecodeError)`.

`portfolio/escalation_router.py:140-146` | P1 | per-reason-portfolio-reload | `_ticker_held` opens both portfolio JSON files per ticker. N reasons → 2N reads. | Fix: cache loaded states in `should_escalate_to_claude` scope.

`portfolio/escalation_router.py:228-251` | P1 | early-return-shadows-criteria | Loop returns on first match per reason. `held_sell_flip` on reason[0] returns before `top5_split` on reason[1] is even checked. Documented as equal-weight criteria but precedence is reason-order-dependent. | Fix: collect all matches, return highest-priority (or document precedence).

`portfolio/regime_alerts.py:30-43, 100-131, 134-146` | P1 | unbounded-jsonl-reads | `_get_last_regime`/`get_regime_distribution`/`get_regime_history` all read full `regime_history.jsonl`. Module is currently unused (no imports outside the file) — latent until wired up. | Fix: use `last_jsonl_entry` with ticker filter or tail-N scan.

`portfolio/agent_invocation.py:947` | P1 | autonomous-first-still-gates-on-window | Same `_is_agent_window` gate; falls back to autonomous so not silent, but the gate name is misleading. | Fix: rename to `_is_us_business_hours` and document semantics.

`portfolio/trigger.py:170-196` | P1 | save-state-prune-contract | `_save_state` relies on caller setting `_current_tickers` for pruning. Silent no-op if absent (only logs WARNING on empty set). | Fix: assert `_current_tickers in state`, log on missing.

`portfolio/agent_invocation.py:1140-1147` | P1 | journal-baseline-vs-prune | `count_jsonl_lines` baseline captured before subprocess; `prune_jsonl` (main.py:360) can run during the subprocess and reduce count_after below count_before → false `journal_written=False`. | Fix: capture baseline AFTER prune, or use ts-based delta as secondary check.

### Lower priority (P2, 70-79)

`portfolio/main.py:282` | P2 | post-cycle-untracked-task | `_maybe_send_digest(config)` not wrapped in `_track` like sibling tasks. Failures don't appear in `report.post_cycle_results`. | Fix: use `_track("digest", _maybe_send_digest, config)`.

`portfolio/main.py:1050` | P2 | safeguard-cycle-id-aliasing | Same restart-reset issue as P0 #4. | Fix: wall-clock gating.

`portfolio/digest.py:168` | P2 | failures-math-approximation | `l2_failures = max(0, invoked - l2_analyses)` doesn't account for autonomous-first `invoked_<why>` statuses or buffered triggers. Approximate but not exact. | Fix: count by (tier, terminal_status) rather than total delta.

`portfolio/escalation_gate.py:160` | P2 | utcnow-deprecated | `datetime.utcnow()` removed in 3.14. | Fix: `datetime.now(UTC)`.

`portfolio/escalation_gate.py:203-215` | P2 | thread-pool-per-call | `ThreadPoolExecutor(max_workers=1)` created and destroyed per gate call. | Fix: module-level singleton executor.

`portfolio/perception_gate.py:65, 79-80` | P2 | gate-blocks-on-empty-signals | Empty signals dict returns False → T1 trigger skipped silently. Recovers next cycle, but no INFO log on the skip path. | Fix: fail-open with WARNING.

`portfolio/market_timing.py:244-260` | P2 | misleading-window-name | `_is_agent_window` returns False on weekends entirely. Name implies general agent layer, actually gates Claude CLI for US-hours only. Confused future maintainers (see P0 #1). | Fix: rename to `_is_us_claude_window`.

`portfolio/agent_invocation.py:1083-1108` | P2 | bat-fallback-prompt-tier-mismatch | Fallback to `pf-agent.bat` is hardcoded T3 but the prompt was built for the requested tier (T1/T2). | Fix: rebuild prompt as T3 when bat-path is used, or skip fallback for non-T3.

`portfolio/multi_agent_layer2.py:188` | P2 | popen-attr-patching | `proc._log_fh = log_fh; proc._name = name` patches user attrs on Popen. Works on CPython, fragile to refactor. | Fix: wrap in dataclass.

`portfolio/multi_agent_layer2.py:228-240` | P2 | post-kill-log-race | Auth scan reads log after `proc.kill()`; truncated logs may miss "Not logged in". Rare. | Fix: not actionable without fsync.

`portfolio/reporting.py:786-810` | P2 | stale-data-fx-mismatch | Stale ticker preserved with prior USD price, no fx_rate snapshot. Layer 2 sees old USD × current FX → SEK drift. | Fix: snapshot fx_rate per ticker in the preservation block.

`portfolio/reporting.py:886-906` | P2 | held-tickers-cache-mid-cycle-stale | Cache keyed on `_run_cycle_id`; mid-cycle portfolio mutation (Layer 2 executes a trade during heartbeat_keepalive) isn't invalidated. | Fix: invalidate on `portfolio_state.json` mtime change.

`portfolio/agent_invocation.py:892-922` | P2 | drawdown-loop-loses-second-context | First-portfolio-breach returns immediately; second portfolio's drawdown not appended to `_drawdown_context`. Observability loss. | Fix: complete the loop before returning.

`portfolio/session_calendar.py:82-89` | P2 | brittle-negative-hour | `utc_hour = cet_hour - offset` produces negative hours for hypothetical early-morning sessions. Currently unreachable but brittle. | Fix: `(cet_hour - offset) % 24` plus date math.

`portfolio/digest.py:222-233` | P2 | exists-then-read-race | `BOLD_STATE_FILE.exists()` then `load_json` — atomic rename window. | Fix: drop the exists() and use load_json default.

`portfolio/loop_processes.py:97-106` | P2 | psutil-info-binding | `info` may be unbound if `process_iter` raises before assignment. Currently safe due to CPython iterator semantics, fragile to refactor. | Fix: init `info = {}` before try.

`portfolio/loop_contract.py:1196-1216` | P2 | tmp-residue-scan-truncation | `rglob('*')` with budget 100 may exhaust budget on archived files before reaching genuine top-level orphans. | Fix: bias toward `DATA_DIR.iterdir()` first, then expand.

`portfolio/trigger.py:284-286` | P2 | trade-detection-consumed-then-buffered | `_check_recent_trade` mutates `last_checked_tx_count` even when trigger is buffered. Trade↔trigger correlation timing loss. Minor. | Fix: defer count update until trigger fires.

`portfolio/autonomous.py:189-254` | P2 | T1-hold-count-includes-triggered | T1 branch counts triggered-but-unheld tickers in `hold_count`; T2 excludes them. Inconsistent reporting. | Fix: align with T2 logic for T1.

`portfolio/agent_invocation.py:763` | P2 | full-jsonl-load-for-auth-check | `load_jsonl(INVOCATIONS_FILE)` reads full file; bounded by prune (5000) but ~5MB per check. | Fix: `load_jsonl_tail(..., 50)`.

`portfolio/agent_invocation.py:1419` | P2 | unnecessary-lock-acquire | `check_agent_completion` always takes `_completion_lock`, even when `_agent_proc is None`. Could short-circuit. | Fix: `if _agent_proc is None: return None` before the lock.

---

## Verification of prior P0/P1s (from 2026-05-19 known-issues list)

| Issue | Status | Evidence |
|---|---|---|
| Layer 2 off-hours skip silently disables decisions (main.py:972-979) | **NOT FIXED** | Same condition still present in non-autonomous-first branch. See P0 #1. |
| Stale reports passed to Layer 2 when agent_summary is old | VERIFIED FIXED | `write_agent_summary` runs at main.py:842 (trigger) and 990 (no-trigger) every cycle. |
| Agent invocation stdin leak | VERIFIED FIXED | `stdin=subprocess.DEVNULL` at agent_invocation.py:1166 and multi_agent_layer2.py:183. |
| Layer 2 kill wedge (FGL P0-12) | VERIFIED FIXED | Force-clear `_agent_proc=None` on taskkill success + wait timeout (line 697-703). |
| Specialist quorum fail (FGL P0-4) | PARTIALLY FIXED | Records `critical_errors` row but early-return semantics broken — see P1 above. |
| Grid fisher stop rearm (2026-05-23 batch 3) | OUT OF SCOPE for this review (grid_fisher module). |

---

## Code coverage summary

| Module | Lines | P0 | P1 | P2 |
|---|---:|---:|---:|---:|
| main.py | 1532 | 2 | 1 | 2 |
| agent_invocation.py | 1724 | 2 | 6 | 5 |
| autonomous.py | 846 | 0 | 1 | 1 |
| trigger.py | 661 | 0 | 3 | 1 |
| trigger_buffer.py | 202 | 0 | 0 | 0 |
| multi_agent_layer2.py | 255 | 0 | 1 | 2 |
| escalation_gate.py | 229 | 0 | 0 | 2 |
| escalation_router.py | 269 | 0 | 2 | 0 |
| market_timing.py | 342 | 0 | 0 | 1 |
| reporting.py | 1361 | 0 | 0 | 2 |
| perception_gate.py | 95 | 0 | 0 | 1 |
| loop_contract.py | 2442 | 1 | 2 | 1 |
| loop_health.py | 235 | 0 | 0 | 0 |
| loop_processes.py | 173 | 0 | 0 | 1 |
| macro_context.py | 403 | 0 | 0 | 0 |
| regime_alerts.py | 217 | 0 | 1 | 0 |
| digest.py | 271 | 0 | 0 | 2 |
| daily_digest.py | 278 | 0 | 0 | 0 |
| weekly_digest.py | 309 | 0 | 0 | 0 |
| session_calendar.py | 211 | 0 | 0 | 1 |
| **Totals** | **12055** | **5** | **17** | **22** |

## Summary

The orchestration subsystem is mature. Two of the five P0s are config-flag latent (no_position_skip default off, specialist_timeout configurable, autonomous_first default off) — they will bite the moment those flags flip. The two truly live P0s are:

- **#1 (off-hours skip)** — Active loss of decision coverage every weekend on 24/7 instruments. This is the canonical "loop runs but produces nothing" failure class the project's own CLAUDE.md warns against (the March-April 2026 auth-silent outage). Same root pattern: subprocess gate returns False with no fallback, no visible alert.
- **#4 (cycle-id restart aliasing)** — IC cache and safeguards effectively disabled under typical restart cadence (auto-restart every code merge per `metals-loop.bat`). Silent data degradation, no alert path.

The remaining P1s are mostly performance (unbounded JSONL reads, repeated portfolio loads) and correctness drifts (status-string mismatches breaking dedup/in-flight checks). None individually catastrophic, but together they erode the contract framework's reliability — and the contract framework is the only line of defence against the auth-silent failure class.

Per CLAUDE.md "System reliability is #1": fix #1 (off-hours fallback) before any other orchestration change. The autonomous fallback already exists and works in the parallel branch; mirroring it into the bare-elif is ~5 lines.
