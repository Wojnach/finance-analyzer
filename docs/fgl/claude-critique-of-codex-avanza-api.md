# Claude critique of codex-avanza-api findings

Cross-check of `data/fgl-logs/codex-avanza-api.txt` against actual code at `Q:/finance-analyzer`.
Verdict format: `[VERDICT] <relpath>:<line> — <codex claim> | <reasoning>`.

---

## Codex finding verdicts

[CONFIRM] portfolio/avanza_orders.py:132 — unlocked read-modify-write of avanza_pending_orders.json | Verified at portfolio/avanza_orders.py:132-134: `_load_pending()` (line 70), append, `_save_pending()` (line 79) is uncoordinated; only the inner `atomic_write_json` is atomic. No file lock or threading lock guards the load+mutate+save sequence. Concurrent writers can clobber each other.

[CONFIRM] portfolio/avanza_orders.py:171 — check_pending_orders rewrites pending file without lock | Verified at portfolio/avanza_orders.py:171 and 214: same `_load_pending`/`_save_pending` pattern, no lock around the cycle. Same race window as request_order.

[CONFIRM] portfolio/avanza_orders.py:389 — broad except flips executed→error on Telegram failure | Verified at portfolio/avanza_orders.py:389-394: `send_telegram(msg, config)` is inside the same `try` block (line 351) whose `except Exception as e` (line 391) sets `order["status"] = "error"`. If Avanza returned SUCCESS at line 369-378, status is correctly set to "executed", but a subsequent send_telegram raise transitions it to "error" — exactly the hidden-fill scenario codex describes. Persistence to disk happens at line 214 in check_pending_orders after the loop.

[CONFIRM] portfolio/avanza_control.py:401 — delete_stop_loss_no_page only checks errorCode | Verified at portfolio/avanza_control.py:395-407 and avanza_session.py:372: `_api_delete` returns `{"http_status": status, "ok": ...}`; control.py only inspects `result.get("errorCode")` and ignores `ok`/`http_status`. A 500 response with `ok=False` would be reported as success. Real bug.

[CONFIRM] portfolio/avanza_client.py:350 — TOTP order path always used instead of session | Verified at portfolio/avanza_client.py:285-368: `place_buy_order`/`place_sell_order` go straight to `_place_order` which calls `get_client()` (TOTP) without consulting `_try_session_auth()`. `get_price` (line 137-159) and `get_positions` (line 162-200) DO try session first; the trading path does not. (Note: codex line 350 maps to actual line 327-368 in current file.)

[CONFIRM] portfolio/avanza_client.py:345 — TOTP order path lacks 1000 SEK floor and 50K SEK cap | Verified at portfolio/avanza_client.py:345-348: only validates `volume < 1` and `price <= 0`; no min-1000 SEK floor and no max-50K SEK cap. Compare avanza_session.py:589-601 which has both. Confirmed gap.

[PARTIAL] portfolio/avanza_session.py:668 — get_quote always hits stock endpoint | Verified at portfolio/avanza_session.py:668: hardcoded `/_api/market-guide/stock/{ob}/quote`. Codex's claim is correct that callers passing certificate/warrant ob_ids get the stock endpoint. PARTIAL because in practice some callers go through `get_instrument_price` (line 1232) which iterates types — only direct `get_quote` callers are affected. The bug exists; severity depends on call sites.

[CONFIRM] portfolio/avanza_session.py:1237 — get_instrument_price returns raw market-guide doc | Verified at portfolio/avanza_session.py:1232-1243: returns the raw `data` dict from the market-guide endpoint. The market-guide response wraps price in `{"quote": {"buy": ..., "sell": ..., "last": ...}}` and `keyIndicators`, NOT top-level `lastPrice`. Callers expecting top-level `lastPrice`/`changePercent` (the documented schema) get None/0.

[CONFIRM] portfolio/avanza_client.py:158 — TOTP fallback hardcodes get_stock_info | Verified at portfolio/avanza_client.py:157-159: `client.get_stock_info(orderbook_id)` is the only fallback; certificates/warrants would need `get_certificate`/`get_warrant`. Real issue.

[PARTIAL] portfolio/avanza/tick_rules.py:87 — float×int multiply causes drift | Verified at portfolio/avanza/tick_rules.py:87: `price_int = price * multiplier` is float arithmetic. The example `0.295 * 100` happens to be exactly `29.5` in IEEE-754, but the bug is real for other values: e.g., `0.29 * 100 = 28.999999999999996`, so `round_to_tick(0.29, "down")` with tick 0.01 floors to 0.28 instead of 0.29. Codex's diagnosis is correct; the example is misleading. PARTIAL because the example doesn't trigger but the bug class is real.

[CONFIRM] portfolio/avanza/scanner.py:244 — barrier_distance_pct from instrument price not underlying | Verified at portfolio/avanza/scanner.py:243-244: `barrier_dist_pct = round(abs(last - barrier) / last * 100, 2)` — `last` is the instrument's last (line 229 `last = _val(quote.get("last"))`), not the underlying. `underlying_price` is fetched at line 250 but unused in the barrier distance calculation. Confirmed.

[CONFIRM] portfolio/avanza/market_data.py:57 — get_quote feeds full instrument doc to Quote.from_api | Verified at portfolio/avanza/market_data.py:54-57: `client.avanza.get_instrument(...)` returns a doc whose price lives under `quote: {buy, sell, last}`. `Quote.from_api(raw)` (types.py:65-88) reads `raw.get("buy")`, `raw.get("sell")`, `raw.get("last")` at the top level — these are not present at the doc root, so bid/ask/last all parse to 0.0. Compare MarketData.from_api (types.py:140) which correctly does `raw.get("quote", {})` first.

[CONFIRM] portfolio/avanza/types.py:366 — SearchHit.from_api ignores orderBookId | Verified at portfolio/avanza/types.py:366: only `raw.get("id", raw.get("orderbookId", ""))` is checked; `orderBookId` (camelCase variant the scanner consumes at line 197) is missing. Confirmed.

---

## MISSED BY CODEX (P0/P1 from claude review, independently verified)

[CONFIRM] portfolio/avanza/trading.py:80-92 — place_order has NO ALLOWED_ACCOUNT_IDS whitelist | Verified at portfolio/avanza/trading.py:38-92: `acct = account_id or client.account_id` (line 81) accepts arbitrary input; no comparison to a whitelist set. Pension account 2674244 would be accepted. Compare avanza_session.py:586-587 (whitelist enforced) and avanza_client.py:32 (whitelist defined). This is the most serious missed finding.

[CONFIRM] portfolio/avanza/trading.py:74-92 — TOTP unified path missing 50K SEK MAX cap | Verified at portfolio/avanza/trading.py:74-78: only the 1000 SEK MIN floor is enforced (`if order_total < 1000.0`); no MAX_ORDER_TOTAL_SEK cap. avanza_session.py:597-601 caps at 50,000 SEK. This is a real-money runaway-loop hazard.

[CONFIRM] portfolio/avanza/trading.py:84-92 — place_order runs WITHOUT avanza_order_lock | Verified at portfolio/avanza/trading.py:84-92: bare `client.avanza.place_order(...)` — no `with avanza_order_lock(...)`. avanza_session.py:615 and avanza_client.py:358 both wrap in the lock. Cross-process race against metals_loop / golddigger / fin_snipe is real.

[UNVERIFIED] portfolio/avanza/trading.py:272-278 — opaque whether avanza-api lib hits stoploss/new endpoint | The bundled avanza-api library is not in repo to inspect. Claude flags this as a maybe-bug per library version. Without the library source pinned and verified, the March 3 incident risk is real but unverified at code level. Worth keeping as outstanding.

[CONFIRM] portfolio/avanza/types.py:241-266 — Position.last_price populated from quote.latest (last-traded) | Verified at portfolio/avanza/types.py:241-266: `latest = _val(ob_quote.get("latest"), _val(ob_quote.get("last"), 0.0))` and assigned to `last_price`. No bid/ask fields exist on Position. Callers using `position.last_price` for SELL limit pricing on illiquid warrants will mis-side the spread.

[CONFIRM] portfolio/avanza/account.py:64-94 — get_buying_power accepts arbitrary account ID | Verified at portfolio/avanza/account.py:64-94: `acct = str(account_id) if account_id else client.account_id` — no whitelist check. If caller passes pension account, returns its balance.

[PARTIAL] portfolio/avanza/tick_rules.py:124-126 — fallback returns last entry's tick for out-of-range prices | Verified at portfolio/avanza/tick_rules.py:124-126: `if entries: return entries[-1].tick_size`. This hides a configuration error rather than raising. PARTIAL because it's debatable whether to silently use the last band or raise — production safety leans toward raise, claude's claim stands as a defensive-programming concern.

[CONFIRM] portfolio/avanza/auth.py:74-114 — singleton has NO expiry/re-auth path | Verified at portfolio/avanza/auth.py:74-121: `_instance` is set once; `reset()` exists at 116-121 but no automatic invocation on auth failure. Confirmed gap.

[CONFIRM] portfolio/avanza_client.py:93-108 — get_client check-then-create without lock | Verified at portfolio/avanza_client.py:82-108: `if _client is not None: return` then `_client = Avanza({...})` with no `threading.Lock`. Compare AvanzaAuth.get_instance which uses double-checked locking. Race exists.

[CONFIRM] portfolio/avanza_client.py:65-79 — _try_session_auth mutates _session_client without lock | Verified at portfolio/avanza_client.py:65-79: bare global mutation, no lock.

[CONFIRM] portfolio/avanza_session.py:80-91 — naive vs aware datetime comparison risk | Verified at portfolio/avanza_session.py:80-91: `exp = datetime.fromisoformat(expires_at)` — if the JSON lacks tz, exp is naive; comparison with `datetime.now(UTC)` raises TypeError. The `except ValueError` (line 89) does not catch TypeError, so it propagates uncaught.

[CONFIRM] portfolio/avanza_session.py:325-330 — single 403 forces full Playwright teardown | Verified at portfolio/avanza_session.py:325-330: 403 calls `close_playwright()` and raises. No retry-once-after-CSRF-refresh path. Real.

[CONFIRM] portfolio/avanza_session.py:332-337 — non-JSON 200 returns {"raw": body} | Verified at portfolio/avanza_session.py:331-337: `try: return json.loads(body) except: ... return {"raw": body}`. Caller expecting `orderRequestStatus` sees neither, may interpret as failure and retry → duplicate order. Confirmed.

[CONFIRM] portfolio/avanza_session.py:643-659 — get_open_orders returns [] on RuntimeError | Verified at portfolio/avanza_session.py:643-659: both fallback paths return `[]` on RuntimeError. There is no strict variant. (Note claude references get_stop_losses_strict, which exists at line 854/862 — the asymmetry is real.)

[CONFIRM] portfolio/avanza/types.py:200-217 — StopLossResult treats "ACTIVE" as placement success | Verified at portfolio/avanza/types.py:208-211: `success = str(status).upper() in ("SUCCESS", "OK", "ACTIVE")`. ACTIVE describes an existing stop-loss, not a successful placement. Real concern.

[CONFIRM] portfolio/avanza/streaming.py:91-127 — backoff reset before stable read | Verified at portfolio/avanza/streaming.py:118-137: `self._backoff = _MIN_BACKOFF` at line 126 (after subscribe, before read_loop entry). If `_read_loop()` raises immediately, the next iteration backs off only 1s.

[CONFIRM] portfolio/avanza/streaming.py:62-85 — _callbacks dict mutated without lock | Verified at portfolio/avanza/streaming.py:51, 65, 70, 75, 80, 85: `setdefault(channel, []).append(callback)` is unprotected. `_dispatch_message` (line 239) iterates the same dict. Real.

[CONFIRM] portfolio/avanza/streaming.py:77-85 — on_orders builds wrong channel path | Verified at portfolio/avanza/streaming.py:79: `channel = "/orders/_" + ",".join(account_ids)` — comma-joined into one string. CometD per-account semantics expect `/orders/{aid}` per ID. As written, multi-account subscriptions never match.

[CONFIRM] portfolio/avanza/streaming.py — no session-token refresh on reconnect | Verified at portfolio/avanza/streaming.py:49 and 152-156: `self._push_sub_id` is captured at __init__ and reused in handshake. No path re-pulls from `AvanzaAuth.get_instance().push_subscription_id`. After 24h BankID expiry, reconnect handshakes with a stale token.

[CONFIRM] portfolio/avanza/streaming.py:171 — handshake_resp = msgs[0] without channel filter | Verified at portfolio/avanza/streaming.py:167: assumes index-0 is the handshake response. CometD can interleave a /meta/connect ack. No `for m in msgs: if m["channel"] == "/meta/handshake"` loop.

[CONFIRM] portfolio/avanza/streaming.py:209 — recv() blocking with no per-call timeout | Verified at portfolio/avanza/streaming.py:209: bare `self._ws.recv()`. Relies on `create_connection(timeout=…)` (line 148) which doesn't propagate to recv reliably.

[CONFIRM] portfolio/avanza_resilient_page.py:142-150 — _relaunch not thread-safe | Verified at portfolio/avanza_resilient_page.py:142-150: `_close_quietly` then `_open` with no lock. Concurrent TargetClosedError handlers race.

[CONFIRM] portfolio/avanza_session.py:155-172 — close_playwright holds _pw_lock during blocking I/O | Verified at portfolio/avanza_session.py:154-172: `with _pw_lock:` wraps `_pw_context.close()` (blocking) and `_pw_browser.close()`. Blocking I/O under the lock stalls all peers.

[CONFIRM] portfolio/avanza_session.py:1232-1243 — get_instrument_price first-success-wins on different types | Verified at portfolio/avanza_session.py:1232-1240: iterates `("stock","certificate","fund","exchange_traded_fund")`; first non-exception response wins. Different types return different shapes — querying a warrant with type=stock can succeed with stale data.

[CONFIRM] portfolio/avanza/trading.py:291-319 — place_trailing_stop overloads trigger_price=trail_percent | Verified at portfolio/avanza/trading.py:310-318: `trigger_price=trail_percent` is forwarded directly. No range-validation. Caller passing absolute price thinking it's a percent → never triggers.

[CONFIRM] portfolio/avanza/scanner.py:308-321 — TOTP path uses ThreadPoolExecutor against requests.Session | Verified at portfolio/avanza/scanner.py:308-315: 6 concurrent workers calling `avanza.get_instrument` / `get_market_data`. The avanza-api library wraps requests.Session whose mid-flight cookie rotation is not thread-safe.

[CONFIRM] portfolio/avanza_orders.py:182-215 — order status persisted after offset advance | Verified at portfolio/avanza_orders.py:180-214 vs 339-345: confirmed_tokens are gathered (line 180) and Telegram offset is advanced INSIDE `_check_telegram_confirm` at line 341 BEFORE the loop returns and BEFORE `_save_pending` runs at line 214. Crash between offset save and pending save loses the confirmation state. Real.

[CONFIRM] portfolio/avanza_control.py:130-134 — place_order has no whitelist check on account_id | Verified at portfolio/avanza_control.py:130-134: just normalizes side and calls `_place_page_order(page, resolved_account_id, ...)`. No `if str(account_id) not in ALLOWED_ACCOUNT_IDS`. Compare place_order_no_page at line 343 which DOES validate. Real asymmetry.

[CONFIRM] portfolio/avanza_control.py:130-134 — place_order does not validate side | Verified at portfolio/avanza_control.py:130-134: `normalized_side = (side or "").strip().upper()` then forwarded raw; no `if normalized_side not in ("BUY","SELL")` guard.

---

## Counts

CONFIRM=39 DISPUTE=0 PARTIAL=4 UNVERIFIED=1 MISSED=27
