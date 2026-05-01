# Plan — Avanza-Specific Adversarial Follow-ups (2026-05-02)

## Scope

Four findings from `docs/PLAN_adversarial_followups_20260502.md` deferred to
this Avanza-focused batch. All touch live trading paths — conservative
wrapping only, preserve existing call signatures.

## Fixes

### 1. P0-4 — `portfolio/avanza_client.py:326` `_place_order` bypasses lock

**Problem.** `_place_order` POSTs to Avanza without holding
`avanza_order_lock`. Concurrent calls (TOTP path used by Telegram CONFIRM
flow + metals_loop holding the page-side lock) can both observe the same
buying_power and overdraw the ISK account.

**Fix.** Wrap the `client.place_order(...)` call in `_place_order` with
`avanza_order_lock(op=f"place_order_totp/{order_type.value}/{orderbook_id}")`.
The op label is distinct from the page-session label so the rate-limit
diagnostic ("which loop hit the busy lock") still works.

`OrderLockBusyError` is allowed to propagate so callers retry next cycle —
matches the existing page-side semantics in
`data/metals_avanza_helpers.place_order`.

`delete_order` (`portfolio/avanza_client.py:364`) is also a mutating call
and gets the same wrap.

### 2. AV-P1-2 — `data/metals_avanza_helpers.py:457-489` `delete_stop_loss` missing lock

**Problem.** Every other mutating helper in `metals_avanza_helpers` (place_order,
place_stop_loss, delete_order) wraps its `page.evaluate` in
`avanza_order_lock`. `delete_stop_loss` is the lone exception — a stop-loss
delete racing against a place_order can leave the position partially
unprotected during the gap.

**Fix.** Wrap the `page.evaluate(...)` in `delete_stop_loss` with
`with avanza_order_lock(op=f"delete_stop_loss/{stop_id}"):` matching the
pattern in `delete_order` and the equivalent function in
`portfolio/avanza_control.delete_stop_loss` (already locked).

### 3. AV-P1-3 — `portfolio/avanza_orders.py:186-198` Telegram CONFIRM not sender-authenticated

**Problem.** `_check_telegram_confirm` already filters by `chat_id` (line
193), but in a Telegram **group** chat anyone the chat owner has admitted
can post a "CONFIRM" message that this loop will execute. The current code
authenticates the chat, not the sender. Even in DM-only deployments, an
attacker who compromised the bot token can deliver fake updates with the
right chat_id.

**Fix.** Add per-sender authentication. Read
`config["telegram"]["allowed_user_id"]` (optional, integer or string).
If set, drop CONFIRM messages whose `msg["from"]["id"]` does not match.
If unset, log a one-time warning at startup and fall back to the existing
chat-only check (don't break existing deployments).

### 4. AV-P1-1 — `scripts/fin_fish_monitor.py:142` wrong DELETE URL

**Problem.** Line 142 calls
`api_delete(f"/_api/trading/stoploss/{stop_id}")` — missing the
`accountId` path segment. The correct shape (per
`portfolio/avanza_session.cancel_stop_loss:911`,
`portfolio/avanza_control.delete_stop_loss:227`,
`data/metals_avanza_helpers.delete_stop_loss:472`) is
`/_api/trading/stoploss/{accountId}/{stopId}`. The buggy URL returns 404
and `delete_stop_loss` quietly returns False, so the monitor leaves stale
stops in place when it tries to retune them on partial fill.

**Fix.** Use the canonical 2-segment path:
`api_delete(f"/_api/trading/stoploss/{account_id}/{stop_id}")`.

## Test plan

- `tests/test_avanza.py`: extend `TestPlaceBuyOrder` / `TestPlaceSellOrder` /
  `TestDeleteOrder` with a thread-safety test confirming concurrent calls
  serialize through the lock.
- `tests/test_metals_avanza_helpers.py`: add `TestDeleteStopLoss` with at
  least one test asserting the lock is acquired around the page.evaluate.
- `tests/test_avanza_orders.py`: extend `TestCheckTelegramConfirm` with
  sender-authenticated tests (allowed user passes, disallowed user dropped,
  missing config falls back to chat-only).
- `tests/test_fin_fish_monitor.py` (new): test the URL composition for the
  delete_stop_loss helper. Mock `api_delete` and assert the path includes
  the account ID.

## Worktree

`/mnt/q/finance-analyzer-avanza-p0p1` on branch
`fix/avanza-p0p1-followups-20260502`.
