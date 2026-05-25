# Self-Review — Cross-Cutting Concerns

Independent pass by main session, in parallel with 8 subsystem reviewers. Focus:
patterns that cross subsystem boundaries (atomic I/O, race conditions, secret
hygiene, headless protocol compliance, doc drift).

## P0 findings

`portfolio/warrant_portfolio.py:198-265` 🔴 **Read-modify-write of `portfolio_state_warrants.json` has no lock.**
`record_warrant_transaction()` loads state, mutates `holdings` + `transactions`, writes back via `save_warrant_state()` — which only delegates to atomic_write_json (atomic write, NOT atomic update). No threading.Lock, no sidecar lock. Contrast with `portfolio_mgr.update_state()` (Patient/Bold) which uses per-file `_state_locks` + `_rotate_backups`. metals_loop, grid_fisher, fin_snipe, and main-loop reporting all read this file; any concurrent writer overwrites a peer's transaction. This recurring P0 was flagged in 2026-05-08, 2026-05-11, 2026-05-13, 2026-05-22, 2026-05-23, 2026-05-24 reviews and remains unfixed. Fix: mirror portfolio_mgr's `update_state(mutate_fn)` pattern, add `_state_locks[warrant_path]`, plus `jsonl_sidecar_lock` if cross-process race exists.

## P1 findings

`.claude/rules/signals.md:1` 🟠 **Rules file drift vs code.** Says `MIN_VOTERS = 3 for all asset classes` but `signal_engine.py:1057` has `MIN_VOTERS_METALS = 2` since 2026-05-11. Says `applicable signal counts: crypto=29, stocks=25, metals=27`; CLAUDE.md says crypto=15/stocks=10/metals=12. Two sources of truth, both stale. Fix: delete rules file or generate it from constants.

`.claude/rules/infrastructure.md:18` 🟠 **T1 timeout drift.** Says `T1 Quick (150s/15 turns)`. agent_invocation.py:191 has `1: {"max_turns": 15, "timeout": 180, ...}`. Code is right (bump from 150→180 on 2026-05-14 was recorded only in inline comment).

`portfolio/agent_invocation.py:48,182,1440` 🟠 **T1 timeout history scattered.** Three different comments mention three different historical values (120, 150, 180). Documentation noise, not a bug — but readers may pick wrong number for new tools. Consolidate into a single comment block at TIER_CONFIG definition.

`portfolio/signal_engine.py:4476` 🟠 **God file: signal_engine.py is 4476 lines.** Single-module surface area for the most consequential subsystem is a review-fatigue trap. Reviewers cannot hold the whole flow in head; logic spread across cluster, voting, gating, regime, persistence, telemetry. Split into `signal_engine/` package: vote_aggregator, accuracy_gate, regime_gate, cluster_dedup, dispatch. Don't ship as a single PR — extract one concern at a time per batch.

`data/metals_loop.py:7880` 🟠 **God file: metals_loop.py is 7880 lines.** Same risk. Embeds 60s loop + 10s silver fast-tick + execution + journal + LLM inference all in one file. Hard to test in isolation; impossible to spot subtle ordering bugs. Extract: `metals_fast_tick`, `metals_executor`, `metals_journal`, `metals_llm_dispatch`.

`portfolio/grid_fisher.py:1970` 🟠 **God file: grid_fisher.py.** Single-process Marja-Folcke-style market-maker logic; tests at this scale are integration-only. Modularize ladder / state / sweep / journal.

`dashboard/app.py:2353` 🟠 **God file: dashboard/app.py.** 52 endpoints in one file; security drift (per agent 8's finding on `export_static.py` re-writing to public /static path) is the predictable failure mode. Move per-domain endpoints into blueprints.

## P2 findings

`portfolio/signal_weights.py:120` 🟡 **120 lines is fine but the file's docstring should explicitly call out the contract — weight ∈ [floor, cap], floor > 0, weight=0 disables vote.** Otherwise future contributors invent semantics.

`docs/SYSTEM_OVERVIEW.md` 🟡 **Module map said to enumerate 142 modules.** Actual count: 287 in portfolio/, 37 in data/, 7 in dashboard/, 75 in scripts/ = 406 module surface. SYSTEM_OVERVIEW is stale by ~2× — readers underestimate review surface.

`tests/` 🟡 **443 test files but TESTING.md mentions ~5994 tests.** Empirical run-counts in CLAUDE.md say "~5,994 tests" — verify by `pytest --collect-only -q | tail -1`. If actual count diverged, accuracy of "26 pre-existing failures" is unverifiable.

## P3 findings

`portfolio/file_utils.py:283` ⚪ **"Unxfails" typo in docstring.** Cosmetic — code is correct.

## Cross-cutting observations

**Atomic I/O compliance.** `json.dump()` direct-write appears only in `file_utils.py` (the atomic primitive itself). Every other module routes through `atomic_write_json`. Strong.

**Subprocess hygiene.** Every `claude` invocation Popen passes `stdin=subprocess.DEVNULL` (agent_invocation:1166, claude_gate:440, multi_agent_layer2:183). The 2026-05-17 P0 from prior FGL is fixed.

**Auth-outage guards.** `_AUTH_ERROR_MARKERS = ("Not logged in", "Please run /login", "Invalid API key")` scanned in `claude_gate.py` after every invocation. The March-April silent-outage failure mode is detectable now. Verify the cooldown gate fires — agent 2 (orchestration) found a P0 where auth failures get logged as `status="timeout"` defeating the cooldown. Cross-reference.

**Secret hygiene.** No `f"...{api_key}..."` formatting into log strings. Telegram tokens redacted by `http_retry._redact_url`. No `print()` calls leaking creds in `portfolio/`. CF-Access JWT verified (no header-trust bypass).

**`CLAUDECODE=` unset.** All 12 bat files that invoke `claude` explicitly unset CLAUDECODE — the Feb-18 34h outage failure mode is prevented.

**Stop-loss API.** All `_api/trading/stoploss/new` calls route through `avanza_session.py:801`. No bypass to regular order API.

**EOD/DST handling.** `grid_fisher.minutes_until_eod()` uses `zoneinfo.ZoneInfo("Europe/Stockholm")` with `astimezone()` — DST-safe.

**Empty-ticker guard.** `signal_engine.py:3152` warns on empty ticker (CRITICAL-2 history). All real callers (main, agent_invocation, backtester) pass non-empty.

**Dashboard auth.** All 52 `/api/*` routes have `@require_auth`. `/logout` exempt is fine. `_get_config` cold-cache + token=None silent-allow is the gap (cross-reference agent 8 finding).

**God-file risk pattern.** Four files >2000 lines: signal_engine (4476), grid_fisher (1970), main (1532), agent_invocation (1724), dashboard/app (2353), metals_loop (7880). These accumulate logic faster than tests can pin it. Plan a documented decomposition order.

## Files reviewed (sampled)

- portfolio/file_utils.py (422 lines)
- portfolio/portfolio_mgr.py (180)
- portfolio/warrant_portfolio.py (~270)
- portfolio/signal_engine.py (~4476, partial — voting and disabled-signal sections)
- portfolio/risk_management.py (~988, drawdown section)
- portfolio/agent_invocation.py (~1724, tier config + Popen sections)
- portfolio/grid_fisher.py (~1970, EOD section)
- portfolio/shared_state.py (~250, cache + dogpile section)
- portfolio/trigger.py (top 60 lines)
- portfolio/subprocess_utils.py (Popen variants)
- portfolio/claude_gate.py (Popen + auth marker section)
- dashboard/auth.py (203, full)
- dashboard/app.py (route enumeration only)
- scripts/win/*.bat (CLAUDECODE unset audit)
