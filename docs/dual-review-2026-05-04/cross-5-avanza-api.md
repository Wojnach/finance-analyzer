# Cross-critique — avanza-api

## Codex findings Claude missed

| Codex finding | Why Claude missed it |
|---|---|
| `avanza_client.py:102-106` — `get_client()` advertises BankID-only mode but unconditionally calls `_load_credentials()` and instantiates `Avanza(...)`. **Most of the facade is unusable without TOTP credentials**, including `place_buy_order`/`place_sell_order` re-exports used by `avanza_orders._execute_confirmed_order()`. | Claude reviewed `avanza_session.py` and the `avanza/` subpackage but missed this contract violation in the facade. The "BankID-only" claim is in comments/docstrings; Codex traced the actual code path. |
| `avanza/market_data.py:55-57` — `client.avanza.get_instrument()` returns market-guide payload with prices nested under `raw["quote"]`. Passing the whole response to `Quote.from_api()` produces `bid/ask/last/high/low = 0.0`. **`portfolio.avanza.get_quote()` returns unusable quotes for normal lookups.** | Claude reviewed `tick_rules.py` thoroughly but didn't audit the quote parsing schema. API-shape bug like the metals_warrant_refresh one. |
| `avanza_control.py:397-404` — `_api_delete()` returns `{"http_status": 403, "ok": false}` on broker failure (does not raise). Wrapper only checks `errorCode` → returns `(True, result)` for 403/500. **Caller proceeds as if stop-loss is gone while it's still active.** | Claude reviewed the order-lock paths but didn't audit `_api_delete()` return shape vs the wrapper interpretation. False-success class of bug. |
| `avanza_session.py:1158-1164` — Re-arm reads `sl["order"]` but snapshot can have either `orderEvent` or `order` shape (`StopLoss.from_api` handles both). **Failed cancel-before-sell can leave position without stop-loss protection.** | Claude noted both shapes exist (in metals-core) but didn't trace the re-arm path here. |
| `avanza_client.py:151-152` — Session-backed `get_price()` returns raw market-guide payload; consumers expect `lastPrice`/`changePercent` at top level. `avanza_tracker.fetch_avanza_prices()` records `0.0` for every instrument under session auth. | Schema mismatch between auth modes. Claude didn't compare TOTP path output to session path output. |
| `avanza_orders.py:389-393` — `try` covers both Avanza order placement AND Telegram notification. If order succeeds but Telegram fails, status overwrites from `executed` to `error` — pending log claims trade failed even though live order exists. **Invites duplicate retries on Telegram outage.** | Claude reviewed avanza_orders for the order lock TOCTOU but missed this status-recording bug. |

## Claude findings Codex missed

| Claude finding | Why Codex missed it |
|---|---|
| `avanza/tick_rules.py:87` — `price_int = price * multiplier` is float multiply despite "integer arithmetic" comment. For `price=1.005, multiplier=1000`: `1004.999...` → off-by-one tick → Avanza rejects orders. | Codex didn't check the IEEE 754 behavior of the multiply. Pure numerical bug. |
| `avanza_session.py:671-712` — `get_positions()` returns positions from ALL accounts including pension (2674244). No filter against `ALLOWED_ACCOUNT_IDS`. `fin_fish.py:1359-1360` calls without post-filtering. Project memory rule `feedback_isk_only.md`. | Codex didn't have access to the project memory file noting "ISK only" restriction. Claude knew to check this. |
| `avanza/trading.py:80-81` — unified-package `place_order()` has NO account whitelist guard. Caller can pass `account_id="2674244"` and trade on pension. | Same project-rule context as above. |
| `avanza/trading.py:105-147` — `modify_order()` skips the 1000-SEK minimum validation (which `place_order()` enforces). | Codex didn't compare modify_order against place_order for symmetric checks. |
| `avanza/auth.py:74-121` — TOTP auth singleton has no expiry detection or re-auth path. After ~24h all TOTP-backed calls throw until process restart. | Codex didn't audit the auth lifecycle. |
| `avanza_orders.py:151-214` — `check_pending_orders()` read-modify-write not cross-process locked. Two processes can both match the same CONFIRM token and both call `_execute_confirmed_order()`. | Claude flagged TOCTOU; Codex didn't. |

## Disagreements

None. **Strongly complementary.** Both reviews caught a class of "schema mismatch / wrong return shape" bugs but in different paths:
- **Codex**: get_instrument quote parsing (`market_data.py`), `_api_delete` ok-flag, stop-loss snapshot shape, session vs TOTP `get_price` shape, place-order facade BankID gate, status overwrite on Telegram failure.
- **Claude**: tick_rules float math, account whitelist gaps, modify_order min-size gap, TOTP no re-auth, order-lock TOCTOU.

## What both missed (likely)

- **`avanza/streaming.py`** — neither flagged anything. Streaming consumer disconnects, message ordering, and unsubscribe paths deserve a focused look.
- **`avanza/scanner.py`** — neither flagged. Scanner uses `tick_rules._cache` (which Claude flagged as unlocked); the scanner's own concurrency model wasn't audited.
- **Cross-account contamination during multi-account refresh** — both flagged account whitelist issues but neither asked whether `account_overview` returns positions in a consistent ordering or whether response pagination can leak.
- **CSRF token rotation under load** — Claude noted CSRF reuse-after-expiry is OK; neither asked about token rotation under concurrent requests.

## Reconciled verdict

**P0 (must fix — money-routing bugs):**
1. **(Codex)** `avanza_control.py:397-404` `_api_delete()` 403/500 reported as success — caller treats stop-loss as cancelled while still live. **Direct money-loss risk.**
2. **(Codex)** `avanza_session.py:1158-1164` re-arm reads only `sl["order"]` shape — failed cancel-before-sell leaves position naked.
3. **(Claude)** `avanza_session.py:671-712` `get_positions()` returns pension account positions with no whitelist filter. **Project rule violation: ISK only.**
4. **(Claude)** `avanza/trading.py:80-81` unified-package `place_order()` no account whitelist — pension account trading possible.
5. **(Codex)** `avanza/market_data.py:55-57` `Quote.from_api()` doesn't unwrap nested `quote` — quotes return zero, downstream consumers see `0.0` everywhere.

**P1:**
6. (Codex) `avanza_client.py:102-106` BankID-only mode broken — `get_client()` requires TOTP creds.
7. (Claude) `avanza/tick_rules.py:87` float multiply causes off-by-one tick → rejected orders.
8. (Codex) `avanza_orders.py:389-393` Telegram failure flips order status to error → duplicate retry risk.
9. (Codex) `avanza_client.py:151-152` session-backed `get_price()` returns wrong shape → tracker records 0.0.
10. (Claude) `avanza/trading.py:105-147` `modify_order()` no min-size guard.

**P2:**
11. (Claude) `avanza_session.py:207-227` 401-inside-recovery leaves stale ctx in caller.
12. (Claude) `avanza/auth.py:74-121` no TOTP session expiry detection.
13. (Claude) `avanza_orders.py:151-214` pending-order check TOCTOU.
14. (Claude) `avanza/tick_rules.py:20, 38-48` `_cache` unlocked iteration.
