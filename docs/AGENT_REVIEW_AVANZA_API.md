# Adversarial Review: avanza-api (Agent Findings)

Reviewer: Code-reviewer subagent (feature-dev:code-reviewer)
Date: 2026-04-08

---

## CRITICAL

### CA1. avanza_orders.py: CONFIRM matches OLDEST pending order, not newest [95% confidence]
**File**: `portfolio/avanza_orders.py:127-131`

Loop iterates `pending` in insertion order (oldest first). When user replies CONFIRM to a
Telegram notification about NVDA, the loop hits a stale SAAB-B order first and executes
THAT instead. Comment says "most recent" but code does "oldest."

**Fix**: Iterate in reverse order or sort by timestamp descending.

### CA2. avanza_orders.py: Telegram offset saved before order status — crash loses CONFIRM [85% confidence]
**File**: `portfolio/avanza_orders.py:118-131`

Telegram offset is advanced (line 199) before order status is persisted (line 139). If
process crashes between these operations, the CONFIRM message is consumed but the order
is still `pending_confirmation`. On next cycle, no CONFIRM exists → order sits in limbo
forever, never confirmed and never expired.

**Fix**: Save order status to disk BEFORE advancing Telegram offset (two-phase commit).

### CA3. avanza_session.py: Playwright context used without lock — race with close [88% confidence]
**File**: `portfolio/avanza_session.py:116-135`

`_get_playwright_context()` acquires `_pw_lock` only during creation. API calls use the
returned context without the lock. `close_playwright()` holds the lock while closing the
browser. Thread A mid-request + Thread B closing browser = exception from closed context
on the trade path with no retry.

### CA4. avanza_session.py: Trailing stop `priceType` set to PERCENTAGE — ambiguous payload [82% confidence]
**File**: `portfolio/avanza_session.py:505-529`

For trailing stops (`FOLLOW_DOWNWARDS`), `stopLossOrderEvent.priceType` is set to
`"PERCENTAGE"` (from `value_type`). The sell price is `0` (market). `priceType=PERCENTAGE`
with `price=0` could be interpreted as "sell at 0% of something" by Avanza backend.

**Fix**: Force `stopLossOrderEvent.priceType = "MONETARY"` for all stop types.

---

## HIGH

### HA1. avanza_client.py: Dynamic ISK account scan could return pension account [83% confidence]
**File**: `portfolio/avanza_client.py:223-248`

`get_account_id()` returns first ISK account found. No exclusion guard for pension account
2674244. If Avanza returns accounts in different order, all orders land on pension.
Inconsistent with `DEFAULT_ACCOUNT_ID = "1625505"` hardcoded elsewhere.

**Fix**: Add explicit exclusion: `if acct_id == "2674244": continue`

### HA2. avanza_client.py: TOTP singleton never reset on expiry — stale auth forever [82% confidence]
**File**: `portfolio/avanza_client.py:55-98`

`_client` cached as module-level singleton. `reset_client()` exists but never called
automatically. When TOTP session expires, all calls return errors but singleton keeps
returning stale client. No recovery path unlike Playwright's 401 handling.

### HA3. avanza_orders.py: CONFIRM path uses TOTP auth, not BankID — different auth state [85% confidence]
**File**: `portfolio/avanza_orders.py:206-222`

Order confirmation imports `place_buy_order`/`place_sell_order` from `avanza_control` →
`avanza_client` (TOTP). Rest of trading uses `avanza_session` (BankID/Playwright). If
only BankID was refreshed (normal flow), TOTP may be expired → CONFIRM execution fails
after user already replied.

### HA4. avanza/types.py: StopLossResult looks for wrong key — stop ID always empty [85% confidence]
**File**: `portfolio/avanza/types.py:212`

`StopLossResult.from_api` checks `stopLossId`, `stop_id`, `id` — but Avanza returns
`stoplossOrderId`. Any caller using `portfolio.avanza.trading.place_stop_loss` gets
`stop_id = ""` → impossible to cancel or track the stop-loss.

**Fix**: Add `"stoplossOrderId"` as first key in lookup chain.

### HA5. avanza_session.py: No session expiry pre-check before trades [82% confidence]
**File**: `portfolio/avanza_session.py:116-135`

After initial context creation, no expiry re-check. ~24h session expires mid-operation →
hard exception during live trade. `is_session_expiring_soon()` exists but isn't called
by `_get_playwright_context`.

---

## MEDIUM

### MA1. avanza_orders.py: Telegram offset file (.txt) with JSON content — legacy confusion [83% confidence]
**File**: `portfolio/avanza_orders.py:154-160`

`avanza_telegram_offset.txt` stores JSON via `atomic_write_json`. Fallback branch reads
as plain text → gets `"{"` → `int()` fails → `offset = 0` → replays all updates from
beginning → could re-fire stale CONFIRM.

### MA2. avanza/streaming.py: Multi-account channel format unverified [80% confidence]
**File**: `portfolio/avanza/streaming.py:77-84`

Comma-joined account IDs in CometD channel may not match Avanza's expected format. Wrong
format: subscription silently succeeds but no messages arrive. Order fill notifications
would be dropped silently.

### MA3. avanza_session.py: get_buying_power fallback may use pension account total [80% confidence]
**File**: `portfolio/avanza_session.py:300-322`

Fallback uses `categories[0].totalValue` — could be pension account depending on ordering.
Would report wrong buying power → potentially oversized orders.

---

## LOW

### LA1. avanza/auth.py: TOTP singleton has no expiry — dead reference after session timeout [80% confidence]
**File**: `portfolio/avanza/auth.py:86-114`

Same family as HA2. `reset()` exists but nothing triggers it on expiry. Auth errors
propagate silently.

---

## Cross-Critique: Claude Direct vs Avanza-API Agent

### Agent found that Claude missed:
1. **CA1**: CONFIRM matches oldest order, not newest — **CRITICAL order execution bug** — total miss
2. **CA2**: Telegram offset/order status race condition — total miss
3. **CA3**: Playwright context race (I noted H12 browser crash recovery but agent found the
   concurrent-access race between API calls and close — a different, worse issue)
4. **CA4**: Trailing stop priceType ambiguity — total miss
5. **HA2**: TOTP singleton never reset — total miss
6. **HA3**: CONFIRM uses different auth path (TOTP vs BankID) — total miss
7. **HA4**: StopLossResult wrong key → empty stop ID — total miss (very impactful)
8. **MA1**: Telegram offset file format confusion — total miss
9. **MA2**: Streaming channel format — total miss

### Claude found that agent confirmed/expanded:
1. **H11**: Session expiry ValueError → agent found broader session lifecycle issues (HA5)
2. **H12**: Playwright context not recovered → agent found concurrent race (CA3, deeper issue)
3. **H13**: No account ID validation → agent confirmed as HA1 with more detail
4. **M8**: api_delete 404-as-success → agent didn't re-raise (different priorities)

### Net assessment:
The avanza-api agent found **4 CRITICAL + 5 HIGH + 3 MEDIUM + 1 LOW = 13 net-new issues**.
CA1 (wrong order confirmed) and HA4 (stop ID always empty) are the most impactful — they
directly affect order execution correctness. The agent's deep reading of avanza_orders.py
uncovered the CONFIRM flow bugs that are invisible without understanding the Telegram
polling + order state machine interaction.

Agent was **dramatically stronger** for this subsystem — the order confirmation bugs are
among the most dangerous findings in the entire review.
