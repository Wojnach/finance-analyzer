# Plan — Last 2 Deferred Adversarial Findings (2026-05-02)

## Context

The 2026-05-02 follow-up batch shipped 5 fixes (commits 0c7fcfe0 through e2332885)
plus the in-flight P1-12 trade-guards gate (519ec6af + f3dbb56f). Two findings
remained explicitly "deferred" because they were tagged architectural / UX:

- **P1-3** Layer 2 bypasses claude_gate (architectural — known design choice).
- **P1-10** avanza_orders.py CONFIRM races (UX change requiring user buy-in).

This plan ships **defensible safe actions** for both — not a full architectural
refactor, but a measurable risk reduction that's testable in isolation.

## Findings — current state

### P1-3: Layer 2 bypasses claude_gate

**Investigation result:** the headline is partially out of date. `agent_invocation.py`
already imports `detect_auth_failure` from `claude_gate` and calls it on agent.log
in `check_agent_completion()` (line 956). However:

1. The detection only runs **after** the agent has already finished (or been killed
   by timeout). If the auth marker appears mid-run via stdout/stderr, it's invisible
   to the loop until completion. (acceptable — auth failures are persistent.)
2. The actual subprocess invocation still uses `subprocess.Popen([claude_cmd, ...])`
   directly (line 626), bypassing claude_gate's:
   - master kill switch (`CLAUDE_ENABLED`)
   - rate-limit warning (`_DAILY_WARN_THRESHOLD`)
   - tree-kill-on-timeout (the gate uses `_run_with_tree_kill`)
   - in-process invocation lock (`_invoke_lock`)
   - invocation log (`claude_invocations.jsonl`)

The "major" gate signal (auth detection) is in fact already wired. The remaining
bypass is design — Layer 2 needs **non-blocking** subprocess (so the loop can keep
running while Claude works), whereas `invoke_claude` is **blocking** with a timeout.

**Defensible safe action — already shipped:** confirm `detect_auth_failure` is
called on the completed agent.log slice (verified at line 956). But there's an
**asymmetry hole**: the auth-error scan only runs on `check_agent_completion()`
when the agent has already exited — it does NOT run on the **timeout-kill path**
(`_kill_overrun_agent` at line 253-327), which simply records `status="timeout"`
and never inspects what the agent printed. If a hung agent printed "Not logged
in" before hanging on a network retry, the failure surfaces as `timeout` not
`auth_error` and never lands in `critical_errors.jsonl`.

**Fix in this batch:**
1. Wire `detect_auth_failure` into the timeout-kill path so we still scan
   captured agent.log output before forgetting about the dead subprocess.
2. Document the design-choice gap in CLAUDE.md / module docstring so future
   sessions don't try to re-route the call through `invoke_claude` (which would
   block the loop).

This is a **strict net-positive safety improvement** with no behavioral change
for the happy path.

### P1-10: avanza_orders.py CONFIRM races

**Investigation result:** the actual race is more concerning than the original
plan note suggested:

- `request_order()` creates a pending order with status `pending_confirmation`.
- `_check_telegram_confirm()` polls Telegram for **any message with text
  "CONFIRM"** in the configured chat (and now the configured user, post AV-P1-3).
- `check_pending_orders()` then matches that single CONFIRM against the **most
  recent pending order** (sorted descending by timestamp).

The races:

1. **Stale-CONFIRM race.** User sends CONFIRM at T=0 for order A. By T+10s,
   order A is processed and a NEW order B has been requested (e.g. by a fast
   metals-loop trigger). Telegram getUpdates polling has a stored offset, so
   the original CONFIRM should NOT replay — but if there's any offset save
   failure (the code suppresses `OSError` on `atomic_write_json` at line 235),
   the offset doesn't advance, the SAME CONFIRM is read NEXT cycle, and it
   confirms order B which the user never authorized. Worse: B can be a different
   instrument or direction.

2. **Wrong-order race.** User intends CONFIRM for order A but order B was
   created **after** A and **before** the user's CONFIRM was received. Sort by
   timestamp DESC means B gets confirmed, not A. The Telegram message that
   prompted the CONFIRM was for A.

3. **No-pending-yet race.** User sends CONFIRM, but before the polling cycle,
   a NEW order C is requested. The CONFIRM (for the previous notification)
   confirms C.

**Defensible safe action — per-order confirmation nonce:**

- `request_order()` generates a 6-character hex nonce (e.g. `a1b2c3`) and
  stores it on the pending order (`order["confirm_token"]`).
- The Telegram notification text becomes `Reply CONFIRM a1b2c3 to execute`
  (instead of plain `CONFIRM`).
- `_check_telegram_confirm()` is rewritten to return either:
  - `None` (no confirmation seen)
  - `str` (the nonce that was matched)
- `check_pending_orders()` confirms the **specific order** whose
  `confirm_token` matches, not the most recent.
- Backwards compat: a bare `CONFIRM` (no token) is rejected with a logged
  warning. This is a deliberate UX break — the user was told the new format
  in the notification. Old pending orders without a `confirm_token` are
  treated as legacy and accept bare `CONFIRM` (so existing pending orders in
  flight when the loop restarts don't get stuck).

This **eliminates the wrong-order races** without requiring any behavior change
the user hasn't already requested (they just type the suggested string from
their own notification).

**Notification call site:** `_execute_confirmed_order` is called by the agent
**after** `request_order` has been invoked. The agent constructs the Telegram
notification string in its own playbook prompt — but to actually surface the
nonce, the system needs to either:
- (a) Have `request_order` itself send the Telegram message (and stop relying
  on the agent), OR
- (b) Return the nonce in the dict so the agent's prompt template can include
  it in the next message it sends.

Path (b) is the smaller change and matches the existing pattern (the agent
already gets the order dict back). For the safe-action fix, we'll choose path
(b) — the nonce is in the returned order dict at key `"confirm_token"`. The
playbook update (telling Layer 2 to include the token in its CONFIRM prompt)
is a docs change (`docs/TRADING_PLAYBOOK.md`).

For deployments where the agent is not the only caller of `request_order`,
`request_order` will additionally log the token at INFO level so any operator
can read it from agent.log if needed.

## What ships in this batch

| # | Finding | What ships | Files | Tests |
|---|---------|------------|-------|-------|
| 1 | P1-3 (Layer 2 bypasses claude_gate) | Wire `detect_auth_failure` into timeout-kill path; document design-choice gap | `portfolio/agent_invocation.py` | `tests/test_agent_invocation.py` |
| 2 | P1-10 (CONFIRM races) | Per-order nonce in `confirm_token`; `_check_telegram_confirm` returns matched token; `check_pending_orders` matches order by token | `portfolio/avanza_orders.py`, `docs/TRADING_PLAYBOOK.md` | `tests/test_avanza_orders.py` |

## What's deliberately NOT in scope

- Routing Layer 2 invocation through `invoke_claude()`. That would block the
  loop on a 600-900s subprocess. Discarded as too risky for a "safe action".
- Backporting CONFIRM nonce to the playbook's existing live prompts beyond a
  one-line addition. The playbook is updated to instruct the agent to read
  `order["confirm_token"]` from the returned dict and include it in the
  Telegram message.

## Test plan

For each fix:
1. Write a regression test in the appropriate `tests/test_*.py` (red).
2. Implement the fix (green).
3. Run the test file in isolation to confirm pass.

After both fixes, run the full test suite parallel to verify no regression:
`.venv/Scripts/python.exe -m pytest tests/ -n auto`.

## Worktree

`/mnt/q/finance-analyzer-last-followups` on branch `fix/last-followups-20260502`.

## Commit cadence

One commit per finding:
- `fix(agent_invocation): P1-3 — auth-scan on Layer 2 timeout-kill path`
- `fix(avanza_orders): P1-10 — per-order CONFIRM nonce eliminates race`

## Outcome (2026-05-02)

| # | Finding | Status | Commit |
|---|---------|--------|--------|
| 1 | P1-3 Layer 2 timeout-kill path missing auth-scan | FIXED + 4 tests + helper extracted, completion-path call site dedupe'd | `5682dca0` |
| 2 | P1-10 CONFIRM races (3 of them) | FIXED + 16 tests added, 12 existing tests adapted | `513645cb` |
| 3 | (Codex-style self-review) CONFIRM word boundary | FIXED + 1 test (`confirmed`/`confirms`/`confirmation` no longer match) | `7e2a4bb9` |

Total: 21 new tests (16 P1-10, 4 P1-3, 1 typo-defense), 12 existing
tests adapted from bool to set return type, 0 production regressions.

### Key design decisions

- **Did not route Layer 2 through `claude_gate.invoke_claude`.** That
  helper is blocking with a timeout; Layer 2 must be non-blocking so the
  60s loop can keep ticking. The MOST important gate signal — auth-failure
  detection — is now wired on every Layer 2 exit path (completion AND
  timeout-kill), which captures the silent-auth-outage failure mode that
  the gate was originally designed for.
- **Did not break existing CONFIRM UX.** Bare CONFIRM still works for
  legacy in-flight orders (those without a `confirm_token` field). New
  orders MUST be confirmed by their specific token, which the user reads
  from the same Telegram message that prompted the confirmation. No new
  out-of-band lookup required.
- **Word-boundary CONFIRM regex.** Discovered during my own code review
  that `confirmed`/`confirms` parse as `confirm` + `ed`/`s` and `ed`
  IS valid hex. Tightened parser to `^confirm(?:\s+|$)` so a chat message
  like "I confirmed by my broker" can never register CONFIRM intent.
- **Token entropy: 24 bits = 16M values.** Birthday-bound for 5 in-flight
  orders is ~7.5e-7 collision rate. 6 hex chars is short enough to type
  on a phone, long enough that typos go to "unknown token" not "wrong
  order".

### Test status

- 149/149 tests pass in touched files (`test_avanza_orders.py`,
  `test_agent_invocation.py`, `test_claude_gate.py`,
  `test_auth_failure_bypass.py`).
- Full suite: 8523 passed, 59 pre-existing xdist isolation failures
  (unrelated; documented in `docs/TESTING.md`). None of the failures
  involve the files I touched.
