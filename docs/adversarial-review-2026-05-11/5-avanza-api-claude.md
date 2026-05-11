# Claude adversarial review: avanza-api

## Summary

The Avanza subsystem has TWO parallel implementations (BankID/Playwright in `avanza_session.py` + TOTP/avanza-api lib in `avanza/`) that disagree on critical invariants. The most dangerous gaps cluster around: (1) order-lock ordering with the Playwright RLock — a metals-loop deadlock waiting to fire, (2) `place_order` paths that never round price to tick, (3) `get_open_orders` swallows the only signal that the orderbook truncated, (4) `verify_default_account` cache poisoning where a single bad ID can be cached forever, and (5) the streaming reconnect that lacks a maximum lifetime and PII redaction. The Codex review will likely catch the obvious lock/tick/pagination items; I focused on cross-module races, log-leak surfaces, and the new account-check whitelist drift since `0d457368`.

## P0 — Blockers

- `portfolio/avanza_session.py:212-232` — Why it bites: `_with_browser_recovery` holds `_pw_lock` (an RLock) across `close_playwright()`, which itself acquires the same RLock recursively, AND then calls `_get_playwright_context()` (lock-acquire) AND then re-invokes `op(ctx)` which may, on the same thread, re-enter via Avanza retries downstream. RLock is reentrant only for the holder; that part is fine. But the `metals_loop` cross-process file lock (`avanza_order_lock`) is acquired BEFORE `api_post` in `_place_order` (line 620), and `api_post` acquires `_pw_lock` inside the same thread. If `_pw_lock` is already held by another worker thread mid-recovery (15-30s relaunch), every other order/cancel waits behind that worker. With `MAX_ORDER_TOTAL_SEK = 50_000` and 2s `OrderLockBusyError`, the in-process lock blocks orders for 30s while the file lock fails after 2s — the next caller throws away the order. Fix: hold `_pw_lock` only for the request itself; do teardown/relaunch outside the lock; or instrument the relaunch to bound `_INITIAL_URL_WAIT_MS=2000` plus `chromium.launch()` to <2s.

- `portfolio/avanza_session.py:608-616` — Why it bites: `_place_order` sends `"price": price` raw, never tick-rounded. Avanza accepts off-tick prices on the API but silently rejects with cryptic `INVALID_PRICE` 30% of the time (see Mar 24 incident). `portfolio/avanza/tick_rules.round_to_tick` exists but is unused by either order path. Combined with `portfolio/avanza/trading.py:84` (also no rounding), 100% of orders bypass tick alignment. Fix: round in both `_place_order` paths before payload assembly, using `round_to_tick(price, ob_id, "down" if side=="BUY" else "up")`.

- `portfolio/avanza_session.py:751` — Why it bites: stop-loss whitelist guard `if acct not in ALLOWED_ACCOUNT_IDS` only protects the session path. The unified package `portfolio/avanza/trading.py:213-288` (`place_stop_loss`) has NO whitelist check — `acct = account_id or client.account_id`. If the TOTP-bound `client.account_id` is overridden (e.g. via a future config update or test-leak), stop-losses can fire on the pension account. Fix: import `ALLOWED_ACCOUNT_IDS` in `portfolio/avanza/trading.py` and add an identical guard on every order/stop-loss/cancel.

- `portfolio/avanza_account_check.py:215-219` — Why it bites: cache poisoning. The cache is keyed only on `(account_id, ok=True)`. If `verify_default_account` is called with `raise_on_mismatch=False` for the FIRST time and somehow returns `ok=True` for a wrong account ID (e.g. operator passes `2674244` from a test), the bad result sticks for the whole process lifetime. `reset_cache()` is the only escape and nothing in production calls it. Worse, `_cache_result` does not include the `category` in the key — if Avanza renames the category mid-day (the very shape drift this module was built to detect), the stale `category="ISK"` lives on. Fix: cache only the success of the specific configured `DEFAULT_ACCOUNT_ID`; refuse `account_id != DEFAULT_ACCOUNT_ID` from poisoning the singleton.

## P1 — High

- `portfolio/avanza_session.py:648-664` — `get_open_orders` swallows the only signal that the orderbook truncated. Quote: `return data.get("orders", data.get("openOrders", []))`. If Avanza paginates (cursor-style — known on accounts with >50 orders), the `nextPageToken` field is ignored. Phantom-order detector in `avanza_tracker` will think those tail orders were silently cancelled. Same hazard in `portfolio/avanza/trading.py:168-185` `get_orders`. Fix: surface a `truncated=True` flag, or follow pagination cursors until exhaustion.

- `portfolio/avanza_account_check.py:71-94` — `_walk_accounts` yields accounts from all three response shapes simultaneously. If Avanza returns BOTH `categorizedAccounts` AND `accounts` (the doc comment in `get_buying_power` says this is the current reality), the same `1625505` is matched in Path A and yields a category like `"INVESTERINGSSPARKONTO"`, then matched again in Path B with category `"ISK"` (different label). The loop breaks on the FIRST hit (line 259) so behavior depends on dict iteration order. After commit `0d457368` `DISALLOWED_CATEGORY_FRAGMENTS` is empty so this currently doesn't bite — but if any fragment is re-added (e.g. for a child-minor account), the verifier becomes non-deterministic. Fix: dedupe by account ID, prefer the most-specific category label.

- `portfolio/avanza_session.py:849-864` — `get_stop_losses` returns `[]` both on read failure AND on legitimately empty result. The "strict" sibling on line 867 fixes this but only for callers that opted in. Any safety-critical path that calls the non-strict variant (grep confirms `get_stop_losses` is still imported in 9 places) makes a sell decision against unknown stop-loss state. Mar 3 incident class. Fix: make the non-strict version raise `RuntimeError` and migrate all callers to use the strict version explicitly when they need the leniency.

- `portfolio/avanza_session.py:1119-1224` — `rearm_stop_losses_from_snapshot` calls `place_stop_loss` per snapshot entry, each of which acquires `avanza_order_lock` (line 800). With 5 stops and 2s timeout, a busy peer in any of the 5 leg-rearms drops the leg — the position ends up partially naked. Worse, the function returns `status="PARTIAL"` so the caller may think they recovered. Fix: hold the order lock once for the whole rearm batch, OR fail the entire batch on first `OrderLockBusyError` and propagate so the caller knows reroll didn't complete.

- `portfolio/avanza/scanner.py:171-188` — `search_query = f"{direction} {query}"` builds a query like `"BULL OLJA"`. The TOTP `search_for_instrument` doesn't tokenize that the way the user expects — it does substring match, so "BULL OLJA" rarely returns BULL oil certs but rather instruments containing the literal string. Then `all_hits` is filtered by `direction in title.upper()` — but `title` may not exist (`hits` use `name`). Quote: `all_hits = [h for h in all_hits if dir_upper in (h.get("title", "") or "").upper()]`. Result: silent zero-result scans. Fix: search by `query` only, then post-filter by both `name` and `title` for direction.

- `portfolio/avanza/streaming.py:118-142` — Exponential backoff has no maximum lifetime or rate-limit awareness. On Avanza CometD HTTP 429 (rate-limited push subscription), the wrapper just reconnects forever, doubling backoff to 60s ceiling. That's fine in steady state, BUT every reconnect issues a fresh `/meta/handshake` with the `subscriptionId` — Avanza throttles handshakes per pushSubscriptionId at ~5/min. The retry loop will trip permanent ban after ~3min of network flakiness. Fix: detect HTTP 1006/1011 close codes from `websocket.recv` and back off to ≥120s; cap total reconnect attempts at 100/hour.

- `portfolio/avanza_orders.py:138-142` — `confirm_token` is logged at INFO level into `agent.log`. Quote: `logger.info("Order requested: %s %dx %s @ %.2f SEK (id=%s, confirm_token=%s)", ...)`. The Trading Playbook says `agent.log` is uploaded to Telegram on errors and is read by background Layer 2 agents. Anyone with file access (or a Telegram error dump including the line) gets a free trade-execution token good for 5min. Fix: log only a prefix/suffix (`confirm_token=ab****`), and document the design intent: the token MUST live only in the request_order return value and the Telegram message body.

## P2 — Medium

- `portfolio/avanza_session.py:1227-1248` — `get_instrument_price` walks four instrument types in sequence and treats ANY exception as "try the next type". A genuine 401 (session expired) will trigger 4× `close_playwright()` cycles before falling through to the orderbook fallback. Each cycle re-launches Chromium (5-15s). Fix: re-raise `AvanzaSessionError` immediately, only swallow `RuntimeError`.

- `portfolio/avanza_session.py:271-291` — `_get_csrf` accepts a stale `ctx` from outside `_pw_lock`. The contract says "from inside an already-locked block" but there is no assert that the caller actually holds the lock. A future refactor that passes `ctx` from a non-locked path silently regresses to the pre-A-AV-1 race.

- `portfolio/avanza_client.py:243-279` — `get_account_id()` caches into `_account_id` after the FIRST whitelisted ISK match. If Avanza's overview reorders accounts mid-process (account opened, closed, renamed), the cached value is never refreshed. `reset_session()` does not reset `_account_id` — only `reset_client()` does — but no callers invoke it on session reauth. Fix: have `reset_client()` and `reset_session()` both clear `_account_id`.

- `portfolio/avanza/tick_rules.py:113-127` — `_find_tick_for_price` fallback returns the LAST entry's tick when price exceeds all ranges. For penny-priced warrants (XBT-TRACKER trades <1 SEK), `entries[-1].tick_size` is the LARGEST tick (often 0.05) — completely wrong for low prices. The walk also doesn't handle the case where `max_price=0` means "no upper bound" if it appears in any non-last entry. Fix: explicit `min_price <= price < max_price` checks with the unbounded sentinel on the last entry only.

- `portfolio/avanza_orders.py:185-216` — Per-cycle iteration calls `_execute_confirmed_order` synchronously inside `check_pending_orders`, which calls `place_buy_order` → `_place_order` → `avanza_order_lock` (2s timeout each). With 5 confirmed orders and a busy metals loop, the main loop blocks 10s. The 60s cycle target is missed if a digest collides. Fix: execute confirmed orders in a worker; or fail-fast after first `OrderLockBusyError` and re-queue the rest.

- `portfolio/avanza/types.py:51-88` — `Quote.from_api` defaults numeric fields to `0.0` on missing data. The downstream effect: a missing `bid` becomes 0.0, downstream `spread = ask - bid = ask` becomes huge, every spread filter thinks the orderbook is illiquid. Mark `bid/ask/last` as `Optional[float] = None` and let callers explicitly handle missing data.

- `portfolio/avanza_session.py:580-606` — `_place_order` rejects `order_total > 50_000 SEK` with a hard `ValueError`. The user's stated risk profile in `memory/feedback_risk_tolerance.md` accepts 10-20% knockout on warrants, and a 200K SEK ISK can legitimately want a 60K SEK leg on a high-conviction signal. The hard ceiling is hardcoded with no config knob. Fix: surface `MAX_ORDER_TOTAL_SEK` as a config setting; default 50K is fine.

## P3 — Low

- `portfolio/avanza/auth.py:38-42` — `_create_avanza_client` catches `Exception` and reraises as `AuthError`, including the original message. If the original exception is a 401 from Avanza with a CSRF/session token leaking in the body, the token ends up in `AuthError.__str__()` and downstream logs. Use `from exc` and don't include the message in the new exception's text.

- `portfolio/avanza_control.py:174-193` — `delete_order_live` uses `page.evaluate(..., [accountId, orderId, token])`. The CSRF token is interpolated into the JS heap and visible to any browser DevTools snapshot. Low surface (headless+local) but if remote debugging is ever enabled, the token leaks. Use `fetch` with credentials and rely on the cookie-jar CSRF, not interpolation.

- `portfolio/avanza/streaming.py:160-180` — `_do_handshake` indexes `msgs[0]` without confirming the handshake message is at index 0. CometD specs allow multi-message responses; the meta/handshake may not be first. Fix: filter by `channel == "/meta/handshake"`.

- `portfolio/avanza/types.py:436-441` — `AccountCash.from_api` falls back from `ownCapital` → `buyingPowerWithoutCredit`. These are different concepts — own capital includes locked-up funds, BPwoCredit is liquid-only. A caller sizing positions off `own_capital` may inflate available cash by 20-40%.

- `portfolio/avanza_session.py:106-120` — `session_remaining_minutes` swallows all exceptions and logs them. If the session file is being atomically replaced by a parallel re-auth, the transient file-not-found ends up as a one-line WARNING. Acceptable, but consider downgrading to DEBUG and counting transient misses.

- `portfolio/avanza_account_check.py:119-138` — `_record_critical_error` writes `account_id` in plaintext to `data/critical_errors.jsonl`. This is intentional per the playbook, but worth flagging: the file is checked into ops scripts and the account number is mildly sensitive. Fix: redact to `162***05` in the message field; keep full ID in `context.account_id` for diagnostics.

## Tests missing

- No test for `place_order` price NOT being tick-rounded — would catch P0 above.
- No test for `_with_browser_recovery` recovery taking longer than the cross-process `avanza_order_lock` timeout (deadlock with metals loop).
- No test for `cancel_all_stop_losses_for` with `get_stop_losses_strict` raising mid-poll (the fail-closed path is asserted but not the timing-sensitive race).
- No test for `verify_default_account` cache poisoning with `raise_on_mismatch=False` after commit `0d457368` widened the allowlist.
- No test for `get_open_orders` truncation/pagination — phantom-cancellation hazard.
- No test for `AvanzaStream` reconnect under repeated 429/1006 close codes — handshake-rate-limit ban scenario.
- No test asserting `confirm_token` is NOT logged at INFO (PII regression guard).
- No test for `_get_csrf(ctx=None)` versus `_get_csrf(ctx=stale)` after a teardown+relaunch race.
- No test for `place_stop_loss` in `portfolio/avanza/trading.py` rejecting an account outside `ALLOWED_ACCOUNT_IDS`.
- No test for tick-rule fallback on prices below the smallest range (warrant penny prices).

## Cross-cut observations

The codebase has two ENTIRELY separate Avanza paths — `avanza_session.py` (Playwright + BankID, used by metals/golddigger/grid_fisher) and `portfolio/avanza/*` (avanza-api lib + TOTP, used by `avanza_control.py` and the dashboard). They duplicate every invariant: `ALLOWED_ACCOUNT_IDS`, `MIN_ORDER_TOTAL = 1000 SEK`, tick rounding, stop-loss endpoint choice. The session path enforces the whitelist on every order/stop; the unified package does NOT. Future changes that touch only one path will diverge silently. Recommend consolidating the whitelist + min-order + tick-rounding into `portfolio/avanza/guards.py` and importing from both order entry points.

The `0d457368` allowlist widening (`DISALLOWED_CATEGORY_FRAGMENTS = ()`) means `verify_default_account` no longer raises on ANY category — it only catches the "account-not-found" case (no overview hit). That is one shape-drift edge case wide of the original intent. The verifier is now essentially a "does Avanza acknowledge my account ID" smoke test, not the category guardrail the docstring implies.

`avanza_order_lock.py` uses `filelock.FileLock` which is a cross-process lock on a real file. The 2s timeout is set per-call but there is no mechanism for the metals loop to know that the grid fisher just acquired the lock for a 30s rearm batch — they'll both think the other is stuck. Consider an in-lockfile JSON header with `{op, acquired_at, pid}` so callers can log meaningful diagnostics on timeout.
