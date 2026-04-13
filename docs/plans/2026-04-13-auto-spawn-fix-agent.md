# 2026-04-13 — Auto-Spawn Fix Agent (Plan Only)

**Status:** DESIGN. Not implemented. Depends on `0706c08` (detect_auth_failure + critical_errors journal) and `ccdea70` (layer2_journal_activity contract).

## Goal

When a `critical_errors.jsonl` entry is written (e.g. `auth_failure`, `contract_violation`), automatically spawn a background Claude Code session to **diagnose and propose a fix** — so the user doesn't need to manually notice and intervene during a live incident.

This extends the existing `_trigger_self_heal` mechanism in `portfolio/loop_contract.py` (which already self-heals on CRITICAL violations), but:

1. Makes the trigger source-of-truth `critical_errors.jsonl` (the single-journal model from the earlier commits), not just contract violations.
2. Hardens the spawner against the failure modes that broke Layer 2 itself (the `--bare` regression).
3. Adds a recursion guard so the fix agent's own failures don't spawn more fix agents.

## The tension

A fix agent is *also* a `claude` subprocess. The exact bug class that necessitated this whole effort — silent `claude` subprocess failure — can break the fix agent too. The design must assume the fix agent will sometimes fail silently, and:

- **never spawn in a loop on repeated failures**
- **never assume success just because exit_code is 0**
- **always log to `critical_errors.jsonl` whether the fix attempt succeeded or failed, so the user sees the accumulated evidence at session start**

## Architecture

```
critical_errors.jsonl (append)
        │
        ▼
fix_agent_dispatcher.py  ─┐ runs every N minutes as a scheduled task
        │                │ (or inline hook in loop_contract._trigger_self_heal)
        ▼                │
  read unresolved entries
        │                │
  ┌─────┴─────┐          │
  │ cooldown? │──yes──► skip (record "throttled" to critical_errors.jsonl)
  └─────┬─────┘          │
        │ no             │
        ▼                │
  ┌──────────────┐       │
  │ recursion    │──yes──► skip + record "recursion_blocked"
  │ guard hit?   │       │
  └──────┬───────┘       │
         │ no            │
         ▼               │
  portfolio.claude_gate.invoke_claude(
     caller="fix_agent_<category>",
     model="opus", max_turns=30,
     prompt=build_fix_prompt(unresolved_entries),
     allowed_tools="Read,Edit,Bash"   # NO Write — fix agent edits, doesn't author
  )
         │
         ▼
  detect_auth_failure already wired in invoke_claude
         │
  ┌──────┴───────────┐
  │ success=False?   │──yes──► record_critical_error(
  └──────┬───────────┘            category="fix_agent_failed", ...)
         │ yes                    increment recursion counter
         ▼
  record "fix_attempt_completed"
  (but: did it actually resolve anything?
   → check whether the original critical_errors entries
     got follow-up "resolution" lines written by the agent.
     If not, the fix was cosmetic; alert.)
```

## Core components

### 1. `scripts/fix_agent_dispatcher.py`

- Reads `data/critical_errors.jsonl`, filters unresolved entries from the last `--hours` (default 24h).
- Groups by `category` so a single fix agent handles related entries.
- Consults `data/fix_agent_state.json` for cooldown + recursion state.
- Calls `portfolio.claude_gate.invoke_claude` directly — re-uses all the detection we already built.
- Writes a `fix_attempt` entry to `critical_errors.jsonl` **before** the call (so even if the dispatcher crashes, we have a record).
- After the call, updates state with success/failure + writes a terminal `fix_attempt_completed` entry.

### 2. State file `data/fix_agent_state.json`

```json
{
  "by_category": {
    "auth_failure": {
      "last_attempt_ts": "2026-04-13T14:00:00Z",
      "last_attempt_success": false,
      "consecutive_failures": 2,
      "blocked_until": "2026-04-13T16:00:00Z"
    }
  },
  "recursion_counter": 0,
  "recursion_window_start": "2026-04-13T14:00:00Z"
}
```

Cooldown rules:
- `SELF_HEAL_COOLDOWN_S = 1800` (30 min) — reuse the existing constant from `loop_contract.py`.
- Exponential backoff on consecutive failures: 30m → 2h → 12h → disabled.
- Recursion guard: if `fix_agent_dispatcher` spawned agents in the last 10 minutes that all produced `fix_agent_failed` entries, refuse new spawns for 1h.

### 3. Prompt design

Critical constraint: the fix agent must be given **enough context** to diagnose without needing the conversation history that led to the failure. Prompt structure:

```
You are the Layer 2 fix agent for finance-analyzer. A critical error was
recorded and not auto-resolved. Your job is to diagnose and propose a fix.

## Unresolved critical errors

<dump relevant lines from critical_errors.jsonl>

## Your instructions

1. Read CLAUDE.md and relevant source files.
2. Identify the root cause.
3. Either:
   a. Make the fix directly with Edit (preferred for simple regressions
      like a re-added --bare flag).
   b. Write the fix to data/proposed_fixes/<timestamp>.md for user review
      when the fix is risky or scope is unclear.
4. When done, append a resolution line to data/critical_errors.jsonl:
   {"ts":"<now>","level":"info","category":"resolution",
    "caller":"fix_agent","resolves_ts":"<original ts>",
    "resolution":"<short description>","message":"...","context":{...}}

Do NOT:
- Modify files outside portfolio/, scripts/, tests/, docs/.
- Kill processes or restart the loop.
- Edit config.json, .env, or credentials.
- Commit or push.

Budget: 30 turns, 15 minutes.
```

Tool allow-list: `Read,Edit,Bash` (no `Write` — forces edits over new-file creation; fewer surprises).

### 4. Scheduler

Two options for invocation:

**Option A — Scheduled task (preferred).** New `PF-FixAgentDispatcher` scheduled task firing every 10 minutes. Runs `scripts/fix_agent_dispatcher.py`, which is a no-op if no unresolved entries exist. Isolated from the loop, no coupling to cycle cadence.

**Option B — Inline in `_trigger_self_heal`.** Direct call from `loop_contract.py` on CRITICAL violation. Tighter latency but couples the loop to the dispatcher's correctness. If the dispatcher has a bug, every CRITICAL violation risks destabilising the loop.

Going with **A**. Keeps the loop unchanged; failure of the dispatcher doesn't cascade.

### 5. Kill switch

A file-based kill switch, analogous to `CLAUDE_ENABLED` in `claude_gate.py`:

- `data/fix_agent.disabled` exists → dispatcher exits immediately, records `disabled` event.
- Rationale: during an outage or experimentation, `touch data/fix_agent.disabled` should instantly stop auto-spawn without editing code or restarting.

## Tests

- `test_fix_agent_dispatcher.py`:
  - No unresolved entries → no spawn, exit 0.
  - Unresolved entry within cooldown → no spawn, records `throttled`.
  - Unresolved entry past cooldown → spawns, records attempt.
  - Kill switch file present → no spawn, records `disabled`.
  - Three consecutive failures → extended cooldown (12h).
  - Recursion guard: if dispatcher was itself invoked by a fix agent (env flag), refuse to spawn.
  - `invoke_claude` returning `status="auth_error"` → `consecutive_failures` increments.

Mock `invoke_claude` — never spawn a real subprocess in tests.

## Batches

1. `scripts/fix_agent_dispatcher.py` + state file management + kill switch. Tests.
2. Scheduled task installer script `scripts/win/install-fix-agent-task.ps1`.
3. Wire a one-line reference into CLAUDE.md STARTUP CHECK: "If critical_errors.jsonl shows a `fix_attempt_failed` entry, the dispatcher has given up — manual investigation required."
4. Document the kill switch in `docs/SYSTEM_HEALTH_CONTRACT.md`.
5. Codex adversarial review.

## Risks

- **Fix agent makes a bad edit.** Allow-list restricts directories; `Edit`-only encourages in-place fixes; we commit the fix via a separate user action, not the agent (no push from the agent).
- **Spawn storm.** Cooldown + exponential backoff + hard recursion cap. Kill switch file as last resort.
- **Fix agent hides the problem.** If the agent writes a bogus `resolution` line, the CLAUDE.md STARTUP CHECK stops surfacing the original error. Mitigation: the dispatcher also writes `fix_attempt_completed` entries that remain unresolved until the user acknowledges them — so even a bogus agent resolution leaves *something* for the user to see.
- **Fix agent requires OAuth but runs under a non-interactive scheduled task.** Already validated — the existing loop's Layer 2 agent runs under the same conditions and `claude_gate.invoke_claude` has been verified working with Max OAuth.
- **Cost / rate limits.** Opus @ 30 turns can burn through daily quota. The cooldown caps attempts at ~48/day worst case. User can tune.

## Explicitly deferred

- Teaching the dispatcher to run `pytest` before considering a fix complete. Nice-to-have; needs sandboxing to avoid running tests against live data.
- Auto-committing + auto-pushing the fix. Strongly avoided — the agent should never push unreviewed code to main on a live trading system.
- Multi-agent fix orchestration (one diagnosis agent + one implementation agent). Single-agent is simpler and matches the existing Layer 2 pattern.
