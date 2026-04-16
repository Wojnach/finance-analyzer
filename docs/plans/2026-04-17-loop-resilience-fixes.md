# Loop Resilience Fixes — multi-day plan (started 2026-04-16 EOD)

## Why

Audit after shipping the accuracy degradation tracker (`feat/accuracy-degradation`,
merged commit `a7c8c08`) surfaced five issues in the Layer 2 invocation
stack. They cluster around a single root cause: the auth-failure detector,
the tier timeout enforcement, and the agent prompt template all assume an
interactive user — but Layer 2 runs headless via `claude -p`. The result
is a self-perpetuating false-positive loop that's been generating
`auth_failure` entries in `data/critical_errors.jsonl` all afternoon.

## Triage (highest leverage first)

### P1 — Auth-failure detector false-positive feedback loop  **SHIP TODAY**

`portfolio/claude_gate.py:151 detect_auth_failure()` greps the agent's
entire post-spawn `agent.log` slice for the literal string `"Not logged in"`.
The protocol in `CLAUDE.md` tells every agent (including Layer 2) to
surface unresolved `critical_errors.jsonl` entries verbatim at session
start. Today's stale auth_failure entries contain the literal `"Not
logged in"` string. So:

1. Agent starts, reads `critical_errors.jsonl`.
2. Agent writes `> claude CLI subprocess printed 'Not logged in'` into its
   chat output.
3. Agent finishes (or times out asking the user — see P1B).
4. `check_agent_completion` reads the agent.log slice, sees `"Not logged in"`,
   logs a NEW `auth_failure` entry.
5. Next agent starts, reads the now-larger `critical_errors.jsonl`,
   re-surfaces ALL of them, generating yet more false positives.
6. Repeat indefinitely.

Today's evidence: `2026-04-16T13:45:45` and `2026-04-16T14:15:01` are both
this echo, not real auth failures (subsequent T2 invocations at 15:25 +
15:45 succeeded with the same OAuth credentials).

**Fix:** Match auth markers ONLY at the start of a line in the first ~16
lines of output. Real Claude CLI auth failures appear as standalone
preamble lines BEFORE any agent turn output. Echoes always appear quoted,
in code blocks, or after substantial agent content. Specifically:

- Reject markers preceded by `` ` ``, `'`, `"`, `(`, `>` on the same line.
- Reject markers found beyond line 16 (CLI auth errors print at the very top).
- Add explicit unit tests for both the real-failure case and the
  echoed-quote case.

**Files:**
- `portfolio/claude_gate.py:151-185` — refactor `detect_auth_failure`.
- `tests/test_claude_gate.py` — add tests `test_detects_real_cli_auth_error`,
  `test_ignores_quoted_marker_in_chat`, `test_ignores_marker_in_code_block`,
  `test_ignores_marker_past_line_16`.

**Verify:** After fix + restart, the next ~5 cycles should produce
zero new `auth_failure` entries from T1 startup invocations. Existing 2
stale entries can then be appended-resolved (P3).

### P1B — T1 hard-timeout not enforced (run on same branch as P1)

`agent_invocation.py:166-208` only enforces tier timeouts when a NEW
invocation is attempted (lazy check in `try_invoke_agent`). Between
invocations, a hung process can run for hours. Today's evidence: T1
invoked 16:04:58 with `timeout=120s`, completed 16:15:01 = **603s**.
Multiplied across 3 stuck T1s today.

**Fix:** Move the wall-clock timeout check from `try_invoke_agent` into
`check_agent_completion`. The latter is called every cycle (60s);
checking elapsed monotonic time and force-killing past timeout closes
the gap. Existing kill logic in `try_invoke_agent` can be extracted into
a helper `_kill_overrun_agent()` and called from both sites.

**Files:**
- `portfolio/agent_invocation.py:166-208` — extract `_kill_overrun_agent`.
- `portfolio/agent_invocation.py:477-494` — call `_kill_overrun_agent`
  before the `poll() is None` early return.
- `tests/test_agent_invocation.py` — add
  `test_check_completion_kills_overrun_agent`.

### P2 — Layer 2 startup quick-checks ask user "How would you like to proceed?"  **SHIP TOMORROW**

When a T1 startup quick-check spawns, Claude reads `CLAUDE.md` →
runs `check_critical_errors.py` → sees unresolved entries → asks
"How would you like to proceed?" Subprocess stdin is dead, so the agent
hits its turn cap (or T1 timeout once P1B is fixed) without doing the
work it was spawned for.

**Fix:** Tier-1 (and tier-2/tier-3) prompt template needs an explicit
"do not block on user input" instruction. Add to `_build_tier_prompt`
in `agent_invocation.py:50-66`:

> If the startup check surfaces unresolved critical errors, log them in
> your journal entry and proceed with the trigger task. Do not ask the
> user — there is no interactive user; you are running as a subprocess
> with no stdin.

**Alternative:** Set an env var `PF_HEADLESS_AGENT=1` in the agent's
process env, and have `CLAUDE.md` check this env var via a conditional
("If PF_HEADLESS_AGENT is set, append unresolved errors to your journal
entry and proceed; do not block").

The env var approach is more robust because CLAUDE.md is the source of
the rule — patching the prompt at every tier boundary forgets it for
new invocation paths (bigbet, iskbets, analyze).

**Files:**
- `portfolio/agent_invocation.py:320-326` — add `PF_HEADLESS_AGENT=1`
  to `agent_env` before Popen.
- `portfolio/bigbet.py`, `portfolio/iskbets.py`, `portfolio/analyze.py`
  — same env injection in their direct subprocess.run() call sites.
- `CLAUDE.md` — add the conditional under the STARTUP CHECK section.
- `tests/test_agent_invocation.py` — assert env var is set on the
  Popen call.

### P2B — Negative duration in critical_errors.jsonl  **SHIP TOMORROW**

The `2026-04-16T13:45:45` entry has `duration_s=-1776254571.5`, which is
`time.time() - time.monotonic()` (or vice versa). The morning fix
`c4b3f45` (BUG-203) standardized `_agent_start = time.monotonic()` and
the elapsed math in `check_agent_completion`, but missed at least one
call site that still uses `time.time()` against the monotonic baseline.

**Fix:** grep for `_agent_start` reads and ensure all use
`time.monotonic()` consistently. Add an invariant test:

```python
def test_duration_s_is_never_negative_on_auth_failure():
    # Spawn fake subprocess, inject auth marker, assert duration_s >= 0.
```

**Files:**
- `portfolio/agent_invocation.py` — audit all `time.time()` /
  `time.monotonic()` mixing. Suspect: `record_critical_error` callsite
  pulling duration from elsewhere.
- `tests/test_agent_invocation.py` — invariant test.

### P3 — Resolve the 2 stale auth_failure entries  **AFTER P1**

After P1 ships and the loop runs ~30 minutes without generating new
false positives, append two resolution entries to
`data/critical_errors.jsonl`:

```json
{"ts":"<now>","level":"info","category":"resolution","caller":"layer2_t1","resolution":"False positive: auth-detector false-positive feedback loop. Fixed in commit <SHA>. Manually resolving prior entries; subsequent cycles produce no new auth_failure entries.","resolves_ts":"2026-04-16T13:45:45.814005+00:00","message":"Resolved by P1 detector fix","context":{}}
{"ts":"<now>","level":"info","category":"resolution","caller":"layer2_t1","resolution":"Same as above — false positive from echo loop. Fixed in commit <SHA>.","resolves_ts":"2026-04-16T14:15:01.612790+00:00","message":"Resolved by P1 detector fix","context":{}}
```

## Execution order

| Day | Work |
|-----|------|
| Today (2026-04-16) | **P1 ONLY** — feedback loop fix + push + restart + verify next 5 cycles produce no new false positives + resolve the 2 stale entries via P3 |
| Tomorrow (2026-04-17) | P1B (T1 timeout) + P2 (env var + CLAUDE.md update) + P2B (duration math audit) — all on one worktree `fix/loop-resilience` since they cluster around `agent_invocation.py` |
| Tomorrow afternoon | Codex post-impl adversarial review of the cumulative branch (the usage limit hit twice today; should be available by then) + any P1/P2 findings + merge + restart |

## /fgl protocol checklist (apply to each branch)

- [x] worktree from main
- [x] read all relevant files first
- [x] write tests with the change
- [x] commit per batch (≤10 files)
- [ ] codex post-impl adversarial review
- [ ] full pytest -n auto
- [ ] merge into main + push via cmd.exe
- [ ] restart loop via `pf-restart.bat loop`
- [ ] tail `data/portfolio.log` to confirm no new error class
- [ ] worktree cleanup

## Recovery — picking up tomorrow

1. `cat docs/plans/2026-04-17-loop-resilience-fixes.md`
2. Check `data/critical_errors.jsonl` — confirm no new false positives
   since P1 shipped + that the 2 manual resolution entries are present.
3. `git log --oneline -5` — confirm P1 commit landed on main and was pushed.
4. Begin P1B in a new worktree: `git worktree add ../finance-analyzer-resilience -b fix/loop-resilience`.
5. Work P1B → P2 → P2B in that order, testing + committing per batch.
6. Codex review the entire branch when done.
7. Merge + push + restart.

## What this does NOT cover

- Snapshot file rotation for `accuracy_snapshots.jsonl` (deferred per the
  degradation tracker plan; not a problem at <5MB after 5 years).
- Per-regime / per-horizon degradation tracking (follow-up after first
  month of snapshot data).
- The 20 pre-existing test failures (per CLAUDE.md known list).
- The chronic BUG-178 ticker pool timeout — already addressed via the
  thundering-herd compute lock shipped this morning (commit `492d478`).
