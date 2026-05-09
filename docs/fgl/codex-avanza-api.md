# Codex Adversarial Review — avanza-api subsystem

Reviewer: codex / gpt-5.4 (xhigh reasoning)
Date: 2026-05-09
Branch: review/2026-05-08-avanza-api (off empty-baseline)

Format: `[Pri] file.py:line — problem | FIX: repair`

---

[P0] portfolio/avanza_orders.py:132 — `request_order()` does an unlocked read-modify-write on `avanza_pending_orders.json`, so concurrent writers can drop or resurrect pending/executed orders even though each individual write is atomic. | FIX: Serialize every pending-order mutation with a cross-process lock or move the state to transactional storage such as SQLite.
[P0] portfolio/avanza_orders.py:171 — `check_pending_orders()` rewrites the same pending-orders file without any lock, so a concurrent `request_order()` save can clobber confirmations or restore stale `pending_confirmation` state. | FIX: Guard the entire load-mutate-save cycle with the same cross-process lock used by every other pending-order writer.
[P0] portfolio/avanza_orders.py:389 — a Telegram send failure after a successful Avanza fill falls into the broad `except` block and flips the order from `executed` to `error`, hiding a live trade and enabling duplicate retries. | FIX: Persist the execution result first and handle notification failures in a separate narrow `try/except` that never overwrites a confirmed fill.
[P0] portfolio/avanza_control.py:401 — `delete_stop_loss_no_page()` only checks for `errorCode`, so the normal `_api_delete()` failure shape (`http_status=500, ok=False`) is reported as success and callers proceed as if the protective stop was removed. | FIX: Base success on `_api_delete()`’s `ok`/`http_status` fields and treat any non-2xx/non-404 response as failure.
[P1] portfolio/avanza_client.py:350 — the shared order path always instantiates the TOTP client instead of using the verified BankID session path, so the documented session-only setup cannot execute confirmed or automated trades at all. | FIX: Dispatch `place_buy_order()`/`place_sell_order()` through `avanza_session` whenever session auth is available and only fall back to TOTP when credentials exist.
[P1] portfolio/avanza_client.py:345 — the TOTP order path validates only sign and volume and skips the 1000 SEK floor plus 50k SEK cap enforced in `avanza_session._place_order()`, so a session outage silently removes order-size risk limits. | FIX: Apply the same minimum/maximum order-total guards in this path before calling `client.place_order()`.
[P1] portfolio/avanza_session.py:668 — `get_quote()` always hits `/_api/market-guide/stock/.../quote`, so certificate/warrant callers like the fish and MSTR tools fetch the wrong endpoint and can trade off missing quotes. | FIX: Resolve the real instrument type first or reuse a type-fallback market-guide lookup instead of hardcoding `stock`.
[P2] portfolio/avanza_session.py:1237 — `get_instrument_price()` returns the raw market-guide document even though callers expect top-level `lastPrice`/`changePercent`, so session-backed consumers silently read `0.0` prices from missing keys. | FIX: Normalize the market-guide payload into the documented flat price schema before returning it.
[P2] portfolio/avanza_client.py:158 — the TOTP fallback hardcodes `get_stock_info(orderbook_id)`, so certificates and warrants lose pricing exactly when the BankID session path is unavailable. | FIX: Route the fallback through an instrument-type-aware lookup or the same market-guide fallback chain used elsewhere.
[P3] portfolio/avanza/tick_rules.py:87 — `round_to_tick()` still multiplies floats before the floor/ceil step, so values like `0.295` can land on the wrong tick despite the “integer arithmetic” claim. | FIX: Convert to an exact integer domain first with `Decimal` or explicit rounding before applying floor/ceil.
[P3] portfolio/avanza/scanner.py:244 — the scanner’s `barrier_distance_pct` is computed from instrument price instead of underlying price, so any future `sort_by="barrier_distance"` ranking will invert knockout risk. | FIX: Measure barrier distance against `underlying_price` or another underlying-level reference rather than `last`.
[P3] portfolio/avanza/market_data.py:57 — this package quote path feeds the full instrument document into `Quote.from_api()`, so real market-guide responses with nested `quote` data parse as zero bids, asks, and lasts. | FIX: Pass `raw.get("quote", raw)` into the parser or flatten the instrument payload before building `Quote`.
[P3] portfolio/avanza/types.py:366 — `SearchHit.from_api()` ignores the `orderBookId` key that the same subsystem’s scanner consumes, so search results can lose the tradable orderbook id. | FIX: Accept `orderBookId` alongside `id` and `orderbookId` when populating `orderbook_id`.
P0=4 P1=3 P2=2 P3=4
