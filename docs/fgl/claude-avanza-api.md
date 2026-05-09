# Adversarial review — avanza-api subsystem

Scope: `portfolio/avanza/` package + `portfolio/avanza_*.py` legacy modules.
All findings are bugs (no praise). Format: `[Pri] file.py:line — problem | FIX: repair`.
COUNT at bottom.

---

## P0 — Real-money loss / wrong account / wrong price

[P0] portfolio/avanza/trading.py:80-102 — `place_order` has NO `ALLOWED_ACCOUNT_IDS` whitelist; caller can pass `account_id="2674244"` (pension) and the order executes. The legacy `avanza_session.py:586-587` and `avanza_client.py:32,270` both enforce the whitelist; this new TOTP-package path silently bypasses it. | FIX: import `ALLOWED_ACCOUNT_IDS = {"1625505"}` and `if str(acct) not in ALLOWED_ACCOUNT_IDS: raise ValueError(...)` BEFORE the API call. Apply to `modify_order`, `cancel_order`, `place_stop_loss`, `place_trailing_stop`, `delete_stop_loss`, `get_buying_power`.

[P0] portfolio/avanza/trading.py:74-92 — no MAX-order guard. `avanza_session.py:597-601` caps single orders at 50,000 SEK to prevent LLM-hallucination / unit-error / runaway-loop full-account drains. The TOTP path has only the 1000 SEK MIN guard. | FIX: copy the `MAX_ORDER_TOTAL_SEK = 50_000.0` cap and `if order_total > MAX_ORDER_TOTAL_SEK: raise ValueError(...)` directly above the `client.avanza.place_order(...)` call.

[P0] portfolio/avanza/trading.py:84-92 — `client.avanza.place_order()` runs WITHOUT `avanza_order_lock`. The cross-process file lock that `avanza_session.py:615`, `avanza_client.py:358`, `avanza_control.py:174` all hold around mutating ops is missing here, so the new TOTP path will race against page-session loops (metals_loop, golddigger, fin_snipe) — exactly the buying-power-snapshot double-spend the lock exists to prevent. | FIX: `with avanza_order_lock(op=f"unified_place_order/{side}/{ob_id}"): client.avanza.place_order(...)`. Same wrap for `modify_order`, `cancel_order`, `place_stop_loss`, `place_trailing_stop`, `delete_stop_loss`.

[P0] portfolio/avanza/trading.py:272-278 — `client.avanza.place_stop_loss_order(...)` — whether the bundled `avanza-api` library posts to `/_api/trading/stoploss/new` (correct, per March 3 incident memo) or the regular order API is OPAQUE from this code. If the lib uses the regular order endpoint, this re-introduces the March 3 instant-fill-at-bad-price bug. MAYBE per library version. | FIX: pin the library version that uses `/_api/trading/stoploss/new`, add an integration test that captures the outgoing URL via mock, and document the pinned version. Until verified, route stop-losses exclusively through `avanza_session.place_stop_loss` (which explicitly hits the safe endpoint at line 796).

[P0] portfolio/avanza/types.py:241-266 — `Position.last_price` is populated from `quote.latest`/`quote.last` (last-traded price), not bid. Callers using `position.last_price` to compute a SELL limit price will hit the wrong side of the spread on every illiquid warrant. The metals warrants we trade routinely have 1–3% spreads. | FIX: split into `bid`, `ask`, `last_price` fields on the dataclass; require callers to pick the side explicitly. Until then, document loudly in the docstring that `last_price` is NOT a sell-able price.

[P0] portfolio/avanza/account.py:64-94 — `get_buying_power(account_id)` accepts arbitrary account IDs, returns the pension account balance if asked for `"2674244"`. A caller using that balance to size an ISK trade will overdraw the ISK. | FIX: `if account_id is not None and str(account_id) not in ALLOWED_ACCOUNT_IDS: raise ValueError(...)` at function entry.

[P0] portfolio/avanza/tick_rules.py:82-98 — the docstring claims "integer arithmetic to avoid floating-point drift", but `price * multiplier` is float×int=float. `0.295 * 100 = 29.4999999...` → `floor(29.499.../1) = 29` → result `0.29` instead of `0.30`. Tick-misaligned price → silent order rejection. Order tools log success but Avanza never accepts the order. | FIX: use `decimal.Decimal` end-to-end, or `int(round(price * multiplier))` to snap to integer domain BEFORE the floor/ceil step, then divide. Add a unit test for `round_to_tick(0.295, "down")` → `0.29` and `round_to_tick(0.295, "up")` → `0.30`.

[P0] portfolio/avanza/tick_rules.py:124-126 — fallback "if price exceeds all ranges, use the last entry's tick" silently returns the wrong tick for out-of-table prices. A 5x silver warrant priced at 500 SEK with a table covering only 0–250 will have the 0.5 SEK tick of the last band applied; if the real Nasdaq tick at 500 is 1.0 SEK, every order is rejected. | FIX: raise `ValueError(f"Price {price} exceeds tick table range")` instead of returning a guessed tick.

[P0] portfolio/avanza/auth.py:74-114 — singleton has NO expiry/re-auth path. `_instance` is set once on first successful TOTP auth; if the underlying `avanza.Avanza` session expires (24h validity for BankID-style sessions on this library), every subsequent call uses a dead client and there is no automatic re-auth. `reset()` exists but no caller invokes it. | FIX: catch `AuthError` / 401 in calling sites OR expose `AvanzaAuth.is_alive()` and call `reset()` + `get_instance(...)` on failure. Compare `avanza_session._with_browser_recovery` (line 207) for the pattern.

---

## P1 — Crash, race, silent corruption, dual-login

[P1] portfolio/avanza_client.py:93-108 — `get_client()` checks `_client is not None` BEFORE acquiring any lock; concurrent first-callers BOTH enter the `Avanza(...)` TOTP authentication path. Avanza rate-limits TOTP attempts and a dual-login burst on startup can temp-lock the account. | FIX: wrap the check + create in a `threading.Lock()` with double-checked locking, mirroring `AvanzaAuth.get_instance` (auth.py:86-114).

[P1] portfolio/avanza_client.py:65-79 — `_try_session_auth()` mutates module-global `_session_client` without lock. Two threads both seeing it as `None` will both call `verify_session()`, doubling cost and creating Playwright contention. | FIX: lock around the read+write.

[P1] portfolio/avanza_session.py:80-91 — `load_session()` calls `datetime.fromisoformat(expires_at)` which returns a naive datetime if the JSON lacks tz info, but compares against `datetime.now(UTC)` (tz-aware). `naive < aware` raises `TypeError` and the `except ValueError` clause does NOT catch it — caller gets a 500-line traceback for what should be a clean "session expired" error. | FIX: `if exp.tzinfo is None: exp = exp.replace(tzinfo=UTC)` before comparison. Same fix at session_remaining_minutes line 110.

[P1] portfolio/avanza_session.py:325-330 — HTTP 403 forces `close_playwright()` and raises `AvanzaSessionError`. CSRF rotation and intermittent edge-cache 403s are both possible mid-session; tearing down the entire Playwright context on a single 403 forces a fresh BankID re-auth and stalls all loops. | FIX: classify 403 as "retry once after CSRF refresh" — re-read `_get_csrf(ctx)`, retry the POST. Only tear down on the second 403.

[P1] portfolio/avanza_session.py:332-337 — successful POST that returns non-JSON (rare but observed at Avanza, e.g. HTML error pages with 200) yields `{"raw": body}`. Caller looking for `orderRequestStatus == "SUCCESS"` sees neither — interprets as failure and retries → DUPLICATE order. | FIX: if `resp.ok` and JSON parse fails, raise a structured exception so caller does NOT silently retry; never return `{"raw": ...}` from a place-order call.

[P1] portfolio/avanza_session.py:643-659 — `get_open_orders()` returns `[]` on `RuntimeError`. Callers using "no open orders → safe to place a new one" make wrong decisions when the API is throwing. | FIX: add `get_open_orders_strict()` that raises, mirror the `get_stop_losses` / `get_stop_losses_strict` split (line 854/862). Pre-place callers MUST use the strict version.

[P1] portfolio/avanza/trading.py:200-217 (types.py StopLossResult.from_api) — `success = status in ("SUCCESS", "OK", "ACTIVE")`. "ACTIVE" is the status of an EXISTING stop-loss observed via `get_stop_losses`, not a successful PLACEMENT. If Avanza ever returns the existing record on duplicate-place attempts, this silently treats it as success even when no new stop was placed. | FIX: drop "ACTIVE" from the success set on placement-result parsing. Use only "SUCCESS"/"OK".

[P1] portfolio/avanza/streaming.py:91-127 — `_run_loop` resets `_backoff = _MIN_BACKOFF` AFTER successful subscribe but BEFORE entering `_read_loop`. If the read loop immediately throws (auth failure on first message), backoff stays at 1s and we busy-loop. | FIX: move the backoff reset INSIDE `_read_loop` after the first message dispatched, OR reset only after N seconds of stable reading.

[P1] portfolio/avanza/streaming.py:50-51,62-85 — `_callbacks` dict mutated by `on_quote`/`on_order_depth`/etc with NO lock. If the user adds a callback after `start()`, `_dispatch_message` (line 239) iterates the same dict — RuntimeError "dictionary changed size during iteration". | FIX: protect with `threading.Lock` OR document "callbacks must be registered BEFORE start() and never after".

[P1] portfolio/avanza/streaming.py:77-85 — `on_orders([acct1, acct2], cb)` registers channel `/orders/_acct1,acct2` (single string), but CometD subscribes per-account: should be `/orders/{account_id}` per ID. As written, no message ever matches. | FIX: iterate the list and register `/orders/{aid}` per account, dispatching to the same callback.

[P1] portfolio/avanza/streaming.py — no session-token refresh. `push_subscription_id` is captured at construction; on 24h BankID expiry (or any session rotation) every reconnect succeeds at the WS layer but the handshake authenticates with a stale token and silently drops. | FIX: pull `push_subscription_id` from `AvanzaAuth.get_instance().push_subscription_id` on every `_do_handshake()` call, OR raise a fatal error so the supervisor restarts the process and re-auths.

[P1] portfolio/avanza/streaming.py:171 — `handshake_resp = msgs[0]` — CometD does not guarantee the handshake response is the first message in the array; can be a delivery/connect ack arriving simultaneously. | FIX: iterate `for m in msgs: if m.get("channel") == "/meta/handshake": handshake_resp = m; break`.

[P1] portfolio/avanza/streaming.py:209 — blocking `recv()` with no per-call timeout; relies on `create_connection`'s `timeout=40s` but the websocket library does NOT propagate that to recv on all platforms. TCP half-open → loop hangs forever, never sending heartbeat, never reconnecting. | FIX: explicit `self._ws.settimeout(_HEARTBEAT_INTERVAL + 5)` after each recv, OR use `select`/`poll` with shorter timeouts and explicit heartbeat scheduling.

[P1] portfolio/avanza_resilient_page.py:142-150 — `_relaunch` not thread-safe. Two threads simultaneously hitting `TargetClosedError` both call `_relaunch`; the second tears down the first's freshly-launched browser. Loop logs "browser dead" repeatedly with no stable session. | FIX: add `threading.Lock` around `_relaunch`, or check `self._page is not None` after re-acquiring lock to skip if a peer already relaunched.

[P1] portfolio/avanza_session.py:155-172 — `close_playwright()` holds `_pw_lock` while `_pw_context.close()` performs blocking I/O (can be 1-3s). Other threads waiting for the lock stall the entire main loop. | FIX: snapshot the references inside the lock, set globals to None, exit the lock, then close the snapshots outside.

[P1] portfolio/avanza_session.py:1232-1243 — `get_instrument_price` tries `("stock","certificate","fund","exchange_traded_fund")` in order; first non-exception response wins. Avanza's market-guide endpoints return DIFFERENT price fields for different instrument types — querying a warrant with `instrument_type="stock"` may return a stale or aggregate price. | FIX: require caller to pass instrument_type, OR validate the response shape (e.g. `keyIndicators.leverage` present → certificate/warrant) before accepting.

[P1] portfolio/avanza_session.py:71-88 (verify_session called inside lock at 184-188) — on transient network blip, `verify_session()` returns False → `close_playwright()` → next call needs full re-launch. Network blips that auto-recover within seconds force unnecessary teardown. | FIX: retry once with 1s backoff before tearing down. Distinguish "auth failed" (401) from "network failed" (timeout, connection refused).

[P1] portfolio/avanza/trading.py:291-319 — `place_trailing_stop(trail_percent, ...)` overloads `trigger_price=trail_percent`. If a caller mistakenly passes the absolute price (e.g. 240.0) thinking it's a percent, the API receives 240% trailing → never triggers, no stop protection. | FIX: validate `0 < trail_percent < 50` at function entry; raise on out-of-range.

[P1] portfolio/avanza/scanner.py:308-321 — TOTP path uses `ThreadPoolExecutor(max_workers=workers)` to fan out `avanza.get_instrument()`/`avanza.get_market_data()` calls. The `avanza-api` library wraps `requests.Session` whose cookie jar mutation during refresh is NOT thread-safe. 6 concurrent calls during a token rotation can corrupt cookies → 401 cascade. MAYBE depending on lib version. | FIX: cap `workers=1` for the TOTP path until the upstream lib guarantees session safety, OR wrap each call in a per-session lock.

[P1] portfolio/avanza_orders.py:182-215 — `for order in pending` mutates `order["status"]` and only persists via `_save_pending(pending)` after the loop. If Telegram offset save (line 341) succeeds but `_save_pending` (line 214) fails (disk full, permission), the offset advances past the CONFIRM, but the order is still "pending_confirmation" → expires on next cycle, user is told it expired despite confirming. Lost confirmation. | FIX: persist order status BEFORE advancing the Telegram offset, or implement a transactional log (append-then-commit) so a crash mid-loop replays cleanly.

[P1] portfolio/avanza_control.py:130-134 — `place_order(page, account_id, ob_id, side, ...)` performs no whitelist check on `account_id`. The caller can bypass `get_account_id()`'s whitelist by passing pension `"2674244"` directly. | FIX: `if str(account_id) not in ALLOWED_ACCOUNT_IDS: raise ValueError(...)` at entry, mirroring `place_order_no_page` at line 343.

[P1] portfolio/avanza_control.py:130-134 — `place_order` does not validate `side`. Caller can pass "BUYY" or "BU" and the page-evaluate forwards it raw — server may default to BUY. Safer path `place_order_no_page` at line 343 DOES validate. | FIX: add the same `if normalized_side not in ("BUY","SELL"): raise ValueError(...)` guard.

---

## P2 — Subtle correctness, fragile parsing

[P2] portfolio/avanza/account.py:42-44 — `positions_raw = raw.get("withOrderbook", raw.get("positions", []))` uses `withOrderbook` first. The raw response has BOTH `withOrderbook` and `withoutOrderbook` (cash positions, FX, etc.). Returning only `withOrderbook` silently drops cash positions from the position list. Callers computing total exposure miss SEK/USD cash. | FIX: merge both: `positions_raw = raw.get("withOrderbook", []) + raw.get("withoutOrderbook", [])`.

[P2] portfolio/avanza/account.py:87-90 — `account.get("accountId", account.get("id", ""))` — order matters. Mid-2026 Avanza migrated category responses to use `id` instead of `accountId` (per the BUG-C7 note in `avanza_session.py:413`). Falling back to `id` AFTER `accountId` means we pick `accountId` if present (fine) but if Avanza removes `accountId` entirely, the previous-cached value is wrong. | FIX: prefer `id` first to match the new shape, fall back to `accountId` for legacy.

[P2] portfolio/avanza/types.py:67-69 — `bid = _val(raw.get("buy"), _val(raw.get("bid"), 0.0))` — if "buy" key exists but its inner value is None, `_val(raw.get("buy"))` returns `None` (default). Outer `_val(...)` then returns the inner default. So far so good. BUT if the "buy" key is `{"value": null}` then `_val` returns the `null` (not the default). `float(None)` → TypeError. | FIX: `_val` should treat `null` value as missing: `if obj.get("value") is None: return default`.

[P2] portfolio/avanza/types.py:285-291 — `Order.from_api` populates `side` from `orderType` first. Some Avanza endpoints use `orderType="STOP_LOSS"` for SL records — caller filtering `[o for o in orders if o.side=="BUY"]` silently misses nothing here, but a caller filtering `if o.side=="SELL"` to count outstanding sell exposure misses stop-losses. | FIX: split: `side` (BUY/SELL) and `order_kind` (NORMAL/STOP_LOSS) on the dataclass; parse separately.

[P2] portfolio/avanza/types.py:209-217 — `StopLossResult` dataclass has no `message` field, so when Avanza returns `{"status":"FAILED","message":"insufficient buying power"}` the caller sees `success=False` but no error reason. Logs at trading.py:280 only print the status. | FIX: add `message: str = ""` to the dataclass; populate from `raw.get("message")` etc.

[P2] portfolio/avanza/scanner.py:188 — direction filter `dir_upper in title.upper()`. "BEAR" matches "TBEAR" (Tencent BEAR product), "REBEARLY" — false positives. | FIX: word-boundary regex: `re.search(rf"\b{dir_upper}\b", title.upper())`.

[P2] portfolio/avanza/scanner.py:259-262 — direction detection iterates `("BULL","BEAR","MINI L","MINI S")`. Order is BULL→BEAR→MINI L→MINI S. A name "MINI S BULL XAG" matches BULL first → classified as BULL despite the "MINI S" (short) prefix. | FIX: check the structural prefix ("MINI L"/"MINI S") FIRST, then fall back to BULL/BEAR substring.

[P2] portfolio/avanza/scanner.py:225-235 — `bid` and `ask` are read from `quote.buy`/`quote.sell` but never validated against the underlying. If Avanza returns `bid=0.001, ask=99` (stale orderbook) the caller computing `spread_pct=(99-0.001)/0.001=9899900%` is filtered out by `max_spread_pct`, fine. But `min_leverage` test at line 327 doesn't exclude these. A bad-tick instrument can still rank top. | FIX: add `if r.bid <= 0 or r.ask <= 0 or r.spread_pct > 25: drop`.

[P2] portfolio/avanza_session.py:1057 — when `poll_read_failed`, `remaining = [sl.get("id","") for sl in initial if sl.get("id") and sl.get("id") not in cancelled]`. Includes empty-string ids if some SLs lacked ids. Then `cancelled = []` at line 1104 wipes the rollback set. Combined with the `remaining` empty-string entries, callers who pass `remaining` to `delete_stop_loss()` would issue a DELETE with empty stop_id. `cancel_stop_loss` line 904 guards empty stop_id, so safe — but the diagnostic logs are noisy. | FIX: filter `if sl.get("id")` before adding to remaining.

[P2] portfolio/avanza_session.py:175-191 — `verify_session()` calls `ctx.request.get(API_BASE + "/_api/position-data/positions")` with no timeout. Default Playwright APIRequestContext timeout is ~30s, which can stack with the 60s loop interval. | FIX: pass `timeout=5000` to the GET.

[P2] portfolio/avanza_session.py:1145-1219 — `rearm_stop_losses_from_snapshot` reads `account = (sl.get("account") or {}).get("id")` and forwards as `account_id` to `place_stop_loss`. If the snapshot's account is the pension account (because cancel_all_stop_losses_for was called without `aid_filter`), re-arm targets pension. | FIX: re-validate against `ALLOWED_ACCOUNT_IDS` before re-arming, refuse with structured error.

[P2] portfolio/avanza/streaming.py:138-142 — `finally` block runs after each iteration of the outer `while`. If the same `_ws` is closed in finally AND in the next iteration's exception handler, the second close logs a "WebSocket already closed" warning under `contextlib.suppress(Exception)` which silences ALL exceptions, hiding real teardown bugs. | FIX: limit the suppress to the specific `WebSocketException` rather than blanket Exception.

[P2] portfolio/avanza/tick_rules.py:19-20 — `_cache: dict[str, list[TickEntry]]` has no thread-safety. `dict[k] = v` is atomic in CPython but reading and reusing a list reference from cache while another thread mutates the list (they can't — entries are produced fresh — but theoretically) is fragile. More importantly, cache is process-lifetime and never invalidated; if Nasdaq updates tick rules mid-session, we keep using stale ticks. | FIX: add an explicit TTL (e.g. 24h) to cached entries; document that callers must call `clear_cache()` after exchange-tick-update events.

[P2] portfolio/avanza_session.py:267-286 — `_get_csrf(ctx=None)` two-path. When `ctx` is None, `_pw_lock` is re-acquired; when called from inside `_op` (line 310) ctx is provided and the lock is NOT re-acquired. Reentrancy was fixed by upgrading to RLock (line 49) so even the no-ctx path is safe — but `_get_csrf(None)` is called from no internal site that I can find, and exposing it externally invites confusion. | FIX: make `ctx` mandatory and remove the no-ctx branch, OR document that the no-ctx branch must never be called from within an existing `_pw_lock` block.

[P2] portfolio/avanza/types.py:35-44 — `_ts(millis)` accepts seconds-resolution unix timestamps (anything >= 0) but interprets them as ms (`millis/1000`). If an upstream endpoint returns seconds (some legacy news feeds do), `1700000000` becomes `1700000.000s = year 1970-01-20`. | FIX: detect by magnitude — `if int(millis) < 1e10: # seconds, multiply *1000`.

[P2] portfolio/avanza/trading.py:213-288 — `place_stop_loss` does not call `cancel_all_stop_losses_for` first. The user memory rule is "Cancel existing stops BEFORE placing a sell" and "sell + stop-loss volume must NOT exceed position size". Placing a new stop while an old one exists can over-encumber the position → `short.sell.not.allowed` error. | FIX: in safety-critical wrappers (especially metals), insist on cancel-then-place. The page-session path in `cancel_all_stop_losses_for` (line 929) handles this; the new TOTP path has no equivalent.

[P2] portfolio/avanza/types.py:333-345 — `StopLoss.from_api` parses `orderEvent` then falls back to `order`. The legacy snapshot format used by `rearm_stop_losses_from_snapshot` (avanza_session.py:1158) reads `sl.get("order")`. Two readers, two key conventions — if Avanza response shape drifts to using only `orderEvent`, `rearm_stop_losses_from_snapshot` breaks silently. | FIX: pick ONE shape, document, fail loudly if both missing.

---

## P3 — Cosmetic / defense in depth

[P3] portfolio/avanza/client.py:19,86 — `DEFAULT_ACCOUNT_ID = "1625505"` is duplicated across `avanza_session.py:35`, `avanza_client.py` (no constant — implicit via `get_account_id`), and `avanza/client.py:19`. A single source of truth is missing. | FIX: move to `portfolio/avanza/_constants.py` and import everywhere.

[P3] portfolio/avanza/__init__.py:34-44 — `__all__` exports both `place_order` (TOTP, no whitelist) and the legacy paths. A casual caller `from portfolio.avanza import place_order` gets the unsafe TOTP version. | FIX: rename the TOTP one to `place_order_unsafe` or simply do not re-export it; make callers go through a vetted facade.

[P3] portfolio/avanza_client.py:127-134 — `find_instrument(query)` calls `client.search_for_stock(query)` which only finds stocks. Misleading name; warrant search returns nothing. | FIX: rename to `find_stock` OR use the union search endpoint.

[P3] portfolio/avanza_session.py:118-127 — default `threshold_minutes=60.0` with a 24h BankID validity means we're "expiring soon" for 23 of the 24 hours. Cosmetic noise in logs/notifications. | FIX: bump threshold to a meaningful warning window (e.g. 120 min).

[P3] portfolio/avanza_session.py:84-88 — "Session expired at {expires_at}" uses the user-facing string. If `expires_at` was tz-naive ISO, the comparison earlier may have succeeded (P1 above) but the message displays a confusing tz-less timestamp. | FIX: format as UTC.

[P3] portfolio/avanza/scanner.py:201 — after `unique_hits[:max_search]`, results are then sliced to `[:limit]` at line 349. The `max_search` cap can silently exceed the user's `limit` if not aligned. Fine, but: `max_search=30` is a magic number applied after dedup; it's the input cap, not the post-dedup cap. | FIX: move the slice AFTER all filters or rename to `pre_dedup_max`.

[P3] portfolio/avanza/types.py:179-197 — `OrderResult.message` flattens a list of error messages with `"; ".join(str(m) for m in message)`. If a message dict has structured fields (e.g. `{"code": 1234, "text": "..."}`) the str(dict) output is unreadable. | FIX: extract `.get("text", str(m))` per element.

[P3] portfolio/avanza_session.py:1100-1104 — when `poll_read_failed`, `cancelled = []` clears the verified-rollback set. The COMMENT (lines 1099-1102) explains it correctly, but the `status="FAILED"` return at line 1094 plus an empty `cancelled` makes the result indistinguishable from "tried nothing". | FIX: add an explicit `read_failed: bool` field to the return dict so callers can branch on the failure mode.

[P3] portfolio/avanza/streaming.py:233-237 — meta-channel filter swallows `/meta/connect` responses including the SUCCESS field. If subscription fails (e.g. 401 on the push endpoint), the failure is silently dropped. | FIX: log meta channel failures at WARNING.

[P3] portfolio/avanza/types.py:418-420 — `OHLC.volume` defaults to `int(...)`. If Avanza returns `2.5e8` as a float for a heavily-traded instrument, `int(2.5e8)` is fine, but if it returns a string `"250000000"`, `int(...)` raises. | FIX: `int(float(raw.get("totalVolumeTraded", 0)))` for safety.

[P3] portfolio/avanza_orders.py:128 — `expires` is computed from `now + timedelta(minutes=EXPIRY_MINUTES)` where `EXPIRY_MINUTES=5`. Hard-coded; not configurable. With the fast-tick metals loop (10s) and main loop (60s), 5min is tight if Telegram has a delivery lag. | FIX: read from config with 5min default.

[P3] portfolio/avanza/account.py:97-149 — `get_transactions(account_id=...)` filters client-side; the docstring even says `Unused by the library call but kept for future server-side filtering`. Each call fetches ALL transactions across all accounts — pension transactions leak through into local filter logic. Privacy-acceptable for own use, but if the filter has a bug, pension data ends up in logs. | FIX: at least document that the upstream library does not filter.

---

## COUNT

- P0: 8
- P1: 22
- P2: 17
- P3: 11
- TOTAL: 58
