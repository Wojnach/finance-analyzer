# Claude Review — avanza-api

## P0 (money-losing or data-corrupting)

(none rated P0; the most severe findings are P1 — wrong-account leakage)

## P1 (high-confidence bugs)

- `portfolio/avanza/tick_rules.py:87` — float multiply contradicts the "integer arithmetic" comment, causes off-by-one tick on certain prices
  ```python
  price_int = price * multiplier   # float * int → still float
  ```
  Comment above (lines 80–84) says "integer arithmetic to avoid float drift" but this is plain float multiplication. For `price=1.005, multiplier=1000`: `1.005 * 1000 = 1004.9999999999999` in IEEE 754. With `tick=0.005`: `1004.9999.../5 = 200.9999...`, floor → 200 → result `1.000` instead of correct `1.005`. **Avanza rejects orders whose prices don't land on the tick grid.** Fix: `price_int = round(price * multiplier)`. Confidence 88.

- `portfolio/avanza_session.py:671-712` — `get_positions()` returns positions from ALL accounts including pension (2674244)
  No filter against `ALLOWED_ACCOUNT_IDS = {"1625505"}`. Caller receives raw union; `account_id` field is set on each position (line 709) but caller has to filter. `portfolio/fin_fish.py:1359-1360` calls this directly without post-filtering. The TOTP path in `avanza_client.get_positions()` (lines 183–199) correctly filters. Memory rule: `feedback_isk_only.md` says only show ISK 1625505. Confidence 90.

- `portfolio/avanza/trading.py:80-81` — unified-package `place_order()` has NO account whitelist guard
  ```python
  client = AvanzaClient.get_instance()
  acct = account_id or client.account_id
  ```
  `avanza_session._place_order()` checks `ALLOWED_ACCOUNT_IDS` at line 586. `avanza_client._place_order()` enforces via `get_account_id()`. The new unified-package `trading.place_order()` does neither. A caller passing `account_id="2674244"` will trade on the pension account. Confidence 85.

- `portfolio/avanza/trading.py:105-147` — `modify_order()` skips the 1000-SEK minimum validation
  `place_order()` raises `ValueError` when `volume * price < 1000` (lines 74–78). `modify_order()` accepts updated price/volume with no equivalent check. Caller can drop an existing order below courtage threshold silently. Confidence 82.

## P2 (concerns / smells)

- `portfolio/avanza/tick_rules.py:20, 38-48, 102-104` — module-level `_cache` dict has no lock; `clear_cache()` races with parallel scanner writes
  Up to 6 scanner threads can read/write `_cache` while `clear_cache()` iterates and clears. Individual dict ops are GIL-protected but iteration during write risks `RuntimeError: dictionary changed size during iteration`.

- `portfolio/avanza/auth.py:74-121` — TOTP auth singleton has no expiry detection or re-auth path
  `get_instance()` creates the singleton once. No heartbeat or `reset()` on auth failure. After ~24h the TOTP session expires; all TOTP-backed calls throw until process restart. Session path has `is_session_expiring_soon()` and `EXPIRY_BUFFER_MINUTES`; TOTP package has none.

- `portfolio/avanza_orders.py:151-214` — `check_pending_orders()` read-modify-write not cross-process locked
  `_load_pending()` / `_save_pending()` are atomic at file level, but the read-modify-write across lines 171–214 is not guarded by `avanza_order_lock`. Two concurrent processes can match the same CONFIRM token and both call `_execute_confirmed_order()`. Broker-level order lock serializes the POST but second call can still run if first unlocked in time.

- `portfolio/avanza_session.py:207-227` — 401 inside `_with_browser_recovery` calls `close_playwright()` mid-recovery
  Local `ctx` reference becomes stale after `_pw_context = None`. Module globals are consistent so next caller is fine, but the in-flight thread holds a stale reference. Practical impact low (Playwright single-threaded by design) but state diverges from caller expectations.

## Did NOT find

1. Wrong stop-loss endpoint — `avanza_session.place_stop_loss()` uses `/_api/trading/stoploss/new` (line 796); unified-package delegates to library that uses the same path.
2. Auth race in `AvanzaAuth.get_instance()` — double-checked locking correctly implemented (lines 86–113).
3. Order lock leak on exception — `avanza_order_lock` uses `try/finally` (lines 95–100), always releases.
4. Streaming disconnect message drop — `_read_loop()` returns cleanly on disconnect; `_run_loop` reconnects.
5. Pension leakage in TOTP `get_positions` path — filtered against `ALLOWED_ACCOUNT_IDS` at lines 183–199.
6. Sub-1000 SEK order accepted via `avanza_session._place_order()` — guarded at line 592.
7. ResilientPage resource leak on error — `_close_quietly()` always tears down before relaunch.
8. CSRF token reused after expiry — `api_post()` reads CSRF from current cookies on every call.
