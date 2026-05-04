# Codex Review — 5-avanza-api

## Summary

The patch contains multiple correctness issues in core flows: the advertised BankID-only mode is not actually supported by several public helpers, quote/price helpers return wrong shapes or zeroed data, and stop-loss/order-state paths can report false success or lose protection. Those are blocking functional bugs rather than nits.

Full review comments:

- [P1] Honor the advertised BankID session path in `get_client()` — Q:\fa-review\portfolio\avanza_client.py:102-106
  In the supported “BankID session only” setup, this function still unconditionally calls `_load_credentials()` and instantiates `Avanza(...)`. That means every caller that goes through `get_client()` — including the `place_buy_order` / `place_sell_order` helpers re-exported from `portfolio.avanza_control` and used by `portfolio.avanza_orders._execute_confirmed_order()` — fails before it ever tries the verified session, so most of this facade is unusable unless TOTP credentials are also configured.

- [P1] Parse quotes from the nested `quote` object returned by `get_instrument()` — Q:\fa-review\portfolio\avanza\market_data.py:55-57
  `client.avanza.get_instrument()` is treated elsewhere in this patch (`scanner.fetch_detail`, `InstrumentInfo.from_api`) as a market-guide payload with prices under `raw["quote"]`. Passing the whole response to `Quote.from_api()` therefore turns bid/ask/last/high/low into `0.0` for normal instrument lookups, so the public `portfolio.avanza.get_quote()` helper returns unusable quotes even when the backend returned real prices.

- [P1] Check `_api_delete()` result before claiming a stop-loss was removed — Q:\fa-review\portfolio\avanza_control.py:397-404
  On the no-page BankID path, `_api_delete()` returns a status dict like `{"http_status": 403, "ok": false}` and does not raise for ordinary broker-side failures. This wrapper only looks for `errorCode`, so a 403/500 response is returned as `(True, result)`. Any caller that relies on the boolean to know the stop is gone can proceed as if volume were freed while the stop-loss is still active.

- [P1] Re-arm stop-loss snapshots that use `orderEvent` — Q:\fa-review\portfolio\avanza_session.py:1158-1164
  The snapshot passed here comes straight from `get_stop_losses_strict()`, and the rest of this patch already treats the sell leg as either `orderEvent` or `order` (`StopLoss.from_api` does exactly that). Reading only `sl["order"]` makes re-arming fail on the `orderEvent` shape, so a failed cancel-before-sell sequence can leave the position without its original stop-loss protection.

- [P2] Normalize the session-backed `get_price()` payload — Q:\fa-review\portfolio\avanza_client.py:151-152
  When BankID session auth is active, `get_instrument_price()` returns the raw market-guide payload, where live values are nested under `quote.*`. Returning that object here violates the documented `lastPrice` / `changePercent` contract, and the in-tree consumer `portfolio.avanza_tracker.fetch_avanza_prices()` therefore records `0.0` for every instrument whenever the session-backed path is used.

- [P2] Keep an executed order marked executed if Telegram notification fails — Q:\fa-review\portfolio\avanza_orders.py:389-393
  This `try` block covers both the Avanza order placement and the Telegram notification. If Avanza accepts the order but `send_telegram()` fails (temporary Telegram outage, rate limit, etc.), the `except` block overwrites `order["status"]` from `executed` to `error`. That leaves the pending-order log claiming the trade failed even though a live order exists, which is exactly the situation that invites duplicate retries.
The patch contains multiple correctness issues in core flows: the advertised BankID-only mode is not actually supported by several public helpers, quote/price helpers return wrong shapes or zeroed data, and stop-loss/order-state paths can report false success or lose protection. Those are blocking functional bugs rather than nits.

## Full review comments

- [P1] Honor the advertised BankID session path in `get_client()` — Q:\fa-review\portfolio\avanza_client.py:102-106
  In the supported “BankID session only” setup, this function still unconditionally calls `_load_credentials()` and instantiates `Avanza(...)`. That means every caller that goes through `get_client()` — including the `place_buy_order` / `place_sell_order` helpers re-exported from `portfolio.avanza_control` and used by `portfolio.avanza_orders._execute_confirmed_order()` — fails before it ever tries the verified session, so most of this facade is unusable unless TOTP credentials are also configured.

- [P1] Parse quotes from the nested `quote` object returned by `get_instrument()` — Q:\fa-review\portfolio\avanza\market_data.py:55-57
  `client.avanza.get_instrument()` is treated elsewhere in this patch (`scanner.fetch_detail`, `InstrumentInfo.from_api`) as a market-guide payload with prices under `raw["quote"]`. Passing the whole response to `Quote.from_api()` therefore turns bid/ask/last/high/low into `0.0` for normal instrument lookups, so the public `portfolio.avanza.get_quote()` helper returns unusable quotes even when the backend returned real prices.

- [P1] Check `_api_delete()` result before claiming a stop-loss was removed — Q:\fa-review\portfolio\avanza_control.py:397-404
  On the no-page BankID path, `_api_delete()` returns a status dict like `{"http_status": 403, "ok": false}` and does not raise for ordinary broker-side failures. This wrapper only looks for `errorCode`, so a 403/500 response is returned as `(True, result)`. Any caller that relies on the boolean to know the stop is gone can proceed as if volume were freed while the stop-loss is still active.

- [P1] Re-arm stop-loss snapshots that use `orderEvent` — Q:\fa-review\portfolio\avanza_session.py:1158-1164
  The snapshot passed here comes straight from `get_stop_losses_strict()`, and the rest of this patch already treats the sell leg as either `orderEvent` or `order` (`StopLoss.from_api` does exactly that). Reading only `sl["order"]` makes re-arming fail on the `orderEvent` shape, so a failed cancel-before-sell sequence can leave the position without its original stop-loss protection.

- [P2] Normalize the session-backed `get_price()` payload — Q:\fa-review\portfolio\avanza_client.py:151-152
  When BankID session auth is active, `get_instrument_price()` returns the raw market-guide payload, where live values are nested under `quote.*`. Returning that object here violates the documented `lastPrice` / `changePercent` contract, and the in-tree consumer `portfolio.avanza_tracker.fetch_avanza_prices()` therefore records `0.0` for every instrument whenever the session-backed path is used.

- [P2] Keep an executed order marked executed if Telegram notification fails — Q:\fa-review\portfolio\avanza_orders.py:389-393
  This `try` block covers both the Avanza order placement and the Telegram notification. If Avanza accepts the order but `send_telegram()` fails (temporary Telegram outage, rate limit, etc.), the `except` block overwrites `order["status"]` from `executed` to `error`. That leaves the pending-order log claiming the trade failed even though a live order exists, which is exactly the situation that invites duplicate retries.
