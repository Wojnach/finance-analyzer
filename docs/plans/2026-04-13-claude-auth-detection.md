# 2026-04-13 — Claude Auth Detection + Silent-Failure Surfacing

**Branch:** `fix/claude-auth-detection`
**Worktree:** `Q:\finance-analyzer\worktrees\claude-auth-detection`

## Root cause

`--bare` added to `claude` subprocess invocations on 2026-03-27 (commit `b4bb57d`) and 2026-04-01 (commit `857fd45`). `--bare` disables OAuth/keychain auth and requires `ANTHROPIC_API_KEY`. User runs Max-subscription OAuth only — no API key. Result: every Layer 2 agent invocation between 2026-03-27 and 2026-04-13 printed `Not logged in — Please run /login` on stdout and exited 0. `claude_invocations.jsonl` recorded `status:"invoked", exit_code:0`. Nothing surfaced the failure. ~3 weeks of silent Layer 2 outage. Multiple Claude sessions ran in that window and none detected it.

## Goals

1. **Fix the immediate bug** — remove `--bare` from all call sites. *(Already applied in this worktree.)*
2. **Detect this class of bug automatically** — scan claude subprocess output for auth-failure markers; escalate to `auth_error` status instead of silently exit-0.
3. **Surface errors aggressively** so *every* future Claude session sees unresolved failures. Not Telegram (user de-prioritised: messages don't always get read). Use a persistent error journal plus a CLAUDE.md-driven startup check.
4. **Add a runtime contract** that catches "Layer 2 enabled but zero journal activity in N hours" so future silent-stall scenarios surface even if we haven't anticipated the failure mode.

## Non-goals (deferred)

- **Auto-spawning a Claude agent to fix detected failures.** User said "ideally" — follow-up. Needs separate design for recursion guards and auth bootstrapping of the spawner.
- **Telegram wiring.** User explicitly de-prioritised.
- **Rewriting `claude_gate.py` architecture.** Out of scope.

## Architecture

```
claude_gate.invoke_claude          (in-process stdout/stderr capture)
  └─ after subprocess: detect_auth_failure(stdout+stderr, caller)
     └─ if True: status="auth_error", critical log, record_critical_error(...)

claude_gate.invoke_claude_text     (in-process stdout capture)
  └─ same treatment

agent_invocation.check_agent_completion   (subprocess stdout → data/agent.log)
  └─ after exit_code known: read agent.log from the offset stored at invoke time
     └─ if auth failure: status override to "auth_error", record_critical_error

multi_agent_layer2.wait_for_specialists   (stdout → data/_specialist_*.log)
  └─ after each proc.wait: read specialist log, detect_auth_failure
     └─ if True: mark specialist failed + record_critical_error
```

Both helpers (`detect_auth_failure`, `record_critical_error`) live in `portfolio/claude_gate.py` since it's already the shared invocation hub.

## `data/critical_errors.jsonl`

Append-only. Each line:

```json
{"ts":"<ISO UTC>","level":"critical","category":"auth_failure",
 "caller":"layer2_t3","marker":"Not logged in",
 "resolution":null,
 "message":"...short summary...",
 "context":{"tier":3,"exit_code":0}}
```

Unbounded file; entries are <500 bytes. Resolutions are recorded as follow-up entries with a matching `resolves_ts` reference rather than mutating earlier entries.

## CLAUDE.md startup surfacing

Append to `Q:\finance-analyzer\CLAUDE.md` a STARTUP CHECK section instructing sessions:

> Before touching the user's task, run `python scripts/check_critical_errors.py`. Any entries from the last 7 days with `resolution: null` mean the system has an unresolved failure that supersedes the user's request — surface it first and ask whether to address it before proceeding.

This leverages the fact that CLAUDE.md is auto-loaded into every session in the project.

`scripts/check_critical_errors.py` prints the unresolved entries in a compact form; returns non-zero exit if any unresolved errors are found (so it's usable in hooks later).

## Runtime contract

Extend `portfolio/loop_contract.py`:

- **Contract: `layer2_journal_activity`**
- **Precondition:** `config.layer2.enabled == true` AND at least one strong trigger has occurred in the last 6 hours (checked via `health_state.json.last_trigger_time`).
- **Check:** `data/layer2_journal.jsonl` has a new entry within 1 hour after the last trigger.
- **On violation:** append to `data/contract_violations.jsonl` (existing mechanism) AND call `record_critical_error(category="contract_violation", ...)` so the same startup surfacing catches it.

Skipping the check when no trigger has occurred avoids false positives on quiet markets.

## Batches

| # | Scope | Files | Tests |
|---|---|---|---|
| 1 | Detection helper + wiring + comments | `claude_gate.py`, `agent_invocation.py`, `multi_agent_layer2.py` | `tests/test_claude_auth_detection.py` (pos/neg markers, e2e mock) |
| 2 | Critical errors journal + check script | `claude_gate.py` (record_critical_error), `scripts/check_critical_errors.py` | `tests/test_critical_errors.py` |
| 3 | CLAUDE.md STARTUP CHECK section | `CLAUDE.md` | n/a (documentation) |
| 4 | Runtime contract `layer2_journal_activity` | `portfolio/loop_contract.py` | `tests/test_loop_contract_layer2.py` |
| 5 | Codex adversarial review + fixes | (review findings) | (any new) |
| 6 | Merge + push + restart loops | (branch merge) | smoke: verify agent.log shows real agent output on next trigger |

After each batch: targeted pytest, then commit. At end of batch 2 and 4: run `pytest tests/ -n auto` to catch regressions.

## Risks

- **False-positive auth detection** — if a benign prompt echoes "Not logged in", we'd downgrade a successful invocation. Markers are narrow enough this is rare; worst case is a noisy error, not data corruption. Acceptable.
- **Log-offset module-global** in `agent_invocation.py` — file already has many module globals (`_agent_proc`, `_agent_log`, `_agent_start`). One more is consistent.
- **Live loop restart** at batch 6 — in-flight agent invocations get killed. Schedule for a non-trigger window.
- **Contract false positives** from timezone skew between `last_trigger_time` and `jsonl` timestamps — verify both are UTC ISO during implementation.

## What is explicitly deferred

- Auto-spawn-fix agent (needs recursion guard + auth bootstrapping design).
- Dashboard surfacing of `critical_errors.jsonl`.
- Unified "system status" artifact that combines `contract_violations.jsonl` + `critical_errors.jsonl`.
