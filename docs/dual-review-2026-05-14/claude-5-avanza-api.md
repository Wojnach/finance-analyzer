# Adversarial Review — 5 avanza-api (main-thread Claude, independent)

## P0 — money-losing or data-corrupting (must fix)

- `portfolio/avanza/trading.py:80-92` — unified-package `place_order` has NO account whitelist check.
  ```python
  client = AvanzaClient.get_instance()
  acct = account_id or client.account_id
  ...
  raw: dict[str, Any] = client.avanza.place_order(
      acct, ob_id, OrderType(side), price, valid, volume,
      condition=Condition(condition),
  )
  ```
  The legacy `avanza_session.place_buy_order/place_sell_order` at line 591 raises if account not in `ALLOWED_ACCOUNT_IDS = {"1625505"}`. Callers migrating to the unified package can pass `account_id=DEFAULT_PENSION="2674244"` (the pension account user explicitly excluded per `feedback_isk_only.md`) and place an order on the wrong account. Same gap for `place_stop_loss`-equivalent in the unified package. Mirror the whitelist guard.

- `portfolio/avanza/account.py:64-94` — `get_buying_power` returns zero-filled `AccountCash` when account is not found in overview.
  ```python
  for account in accounts:
      if str(account.get("accountId", account.get("id", ""))) == acct:
          return AccountCash.from_api(account)
  logger.warning("get_buying_power account_id=%s not found in overview", acct)
  return AccountCash(buying_power=0.0, total_value=0.0, own_capital=0.0)
  ```
  Grid fisher's `GRID_BUYING_POWER_STALE_GRACE_SECS=300` policy says "if we can't read live BP we fail-closed". But `get_buying_power` doesn't FAIL — it returns 0.0, indistinguishable from a real "no cash" reading. Grid fisher's cache then sees 0.0 as a successful reading and refreshes the cache to 0.0; the next 300-second window blocks ALL placements (false negative). Worse, a caller that interprets 0.0 as "skip sizing check" and proceeds to place → places on whatever account_id was passed (combine with P0 above). Raise an exception instead of returning zeroes.

- `portfolio/avanza/trading.py:65-78` — order total guard at 1000 SEK applies to LIMIT orders but `cancel_order` and `modify_order` don't carry it. If a caller modifies a 1500 SEK leg to 200 SEK qty, the resulting 200 SEK leg passes (modify path skips the >=1000 SEK check). Then place_order's check has no effect on already-modified legs.

## P1 — high-confidence bugs (should fix)

- `portfolio/avanza_session.py:83-95` — session expiry check silently proceeds when `expires_at` is unparseable.
  ```python
  try:
      exp = datetime.fromisoformat(expires_at)
      ...
  except ValueError:
      logger.warning("Cannot parse expires_at %r — cannot verify expiry, proceeding with caution", expires_at)
  ```
  "Proceeding with caution" means "ignoring the expiry check and making the API call anyway". If Avanza ever changes the timestamp format (Z-suffix instead of +00:00, microseconds, ISO 8601 variant), every call after that quietly bypasses expiry validation and the next 401 is the first warning operators see. Also: `EXPIRY_BUFFER_MINUTES = 30` constant defined at line 32 is dead code — search shows it's never referenced in the expiry comparison. Fail closed on parse error.

- `portfolio/avanza_session.py:720-811` vs `portfolio/avanza/trading.py:84` — two parallel order-placement paths exist. Callers can pick either; they have DIFFERENT safety guarantees:
  - `avanza_session.place_buy_order/place_sell_order/place_stop_loss`: ALLOWED_ACCOUNT_IDS check, MAX_ORDER ceiling check, leg-size warning, `avanza_order_lock` cross-process lock.
  - `portfolio/avanza/trading.place_order`: order_total >= 1000 SEK check, NO whitelist, NO lock, NO MAX_ORDER ceiling.

  This asymmetry is a P1 hazard: any code change that switches a call from legacy to unified silently drops two safety nets. Consolidate or document the safety contract on every caller.

- `portfolio/avanza/account.py:74-90` — `get_overview_raw()` is called once per `get_buying_power`. If `get_buying_power` is invoked N times across N instruments in a single cycle (grid_fisher does this for 3 metals + 3 oil legs), we hit `/_api/account-overview` 6 times per minute. Avanza rate-limits this. `grid_fisher_config.GRID_BUYING_POWER_CACHE_SECS=60` is the right answer but only used inside grid_fisher; the unified package's account.py itself doesn't cache.

- `portfolio/avanza_session.py:759-762` — non-trailing MONETARY stop must have sell_price > 0. But for `LESS_OR_EQUAL` trigger with `value_type="MONETARY"`, the system raises ValueError. Caller in metals_loop or fish_engine that constructs `sell_price = max(bid * 0.99, 0.01)` — if bid is 0, the stop placement raises and the position is left without a stop. Coordinate with grid_fisher's P0-4 floor logic so the failure mode is "skip & retry" not "exception".

- `portfolio/avanza_session.py:768-775` — `if leg_total < 1000.0: logger.warning(...)` for sub-1000 SEK stop legs. Comment says "metals_loop callers can legitimately produce sub-1000 legs". The warning is necessary but not sufficient — operators can't filter warnings from genuine bugs in logs. Add a structured metric so the dashboard can show cumulative sub-1000 legs / day; current code surfaces nothing.

- `portfolio/avanza/account.py:139-140` — `if account_id is not None: transactions = [t for t in transactions if t.account_id == str(account_id)]`. Filter is client-side AFTER fetching ALL transactions (potentially across both ISK and pension). If we always want ISK-only, pre-filter via API params if available; otherwise the pension account's transaction history leaks into our analysis caches.

- `portfolio/avanza/scanner.py` — per the subagent finding, BankID path ignores `itype_str` filter. Risk: scanner returns OFFERS or RIGHTS issues instead of warrants. Verify on next run.

## P2 — concerns / smells (worth addressing)

- `portfolio/avanza_session.py:35-43` — `DEFAULT_ACCOUNT_ID = "1625505"` and `ALLOWED_ACCOUNT_IDS = {DEFAULT_ACCOUNT_ID}` defined at module level. A future caller importing only one of these silently loses the link. Use a frozenset declared in one place.

- `portfolio/avanza_session.py:800` — `avanza_order_lock(op=f"place_stop_loss/{orderbook_id}")`. Lock scope is per-orderbook; if two different ob_ids stop-loss simultaneously, the locks don't coordinate. For warrants this is fine (each ob has its own state on Avanza side). But documentation should clarify the contract.

- `portfolio/avanza/trading.py:160-165` — `cancel_order` returns bool. Compare:
  ```python
  success = status == "SUCCESS"
  return success
  ```
  Avanza's documented success values include `"SUCCESS"`, `"ACCEPTED"`, `"OK"` depending on endpoint. Only `"SUCCESS"` is recognized here. If Avanza returns `"ACCEPTED"` for a cancel, caller sees False, retries the cancel, may double-cancel an already-canceled order. Mostly harmless but the retry storm adds noise to logs.

- `portfolio/avanza/streaming.py` — per subagent, channel built from raw account IDs with no whitelist check. Real-time market data subscription doesn't transmit orders, so blast radius is small (information disclosure of pension account positions to local subscriber). Still worth a whitelist on subscription target.

- `portfolio/avanza_account_check.py` (new file 2026-05-11 per subagent context) — `DISALLOWED_CATEGORY_FRAGMENTS = []` per grid_fisher_config comment. Empty disallow list = anything allowed. Worth a explicit ISK-only category check.

- `portfolio/avanza_session.py:752` — `valid_until = (date.today() + timedelta(days=valid_days)).isoformat()` with default `valid_days=8`. Stop-losses live for 8 days unless explicitly extended. After a Friday afternoon stop placement, next Sunday is day 9 → expires before Monday open. Bump default to 14 or align to next-Tuesday rollover. Minor.

- `portfolio/avanza_session.py:84` — `expires_at = data.get("expires_at")`. If the key is missing entirely (Avanza API schema change), the entire expiry check at line 85 is skipped (truthy guard), and we proceed with un-checked session. Same fail-open as the unparseable-format case.

## Did NOT find

1. **Silent failures**: see P0/P1 above. expires_at parse failure (P1).
2. **Race conditions**: avanza_order_lock used for stop-loss placement; per-ob scope is correct.
3. **Money-losing bugs**: P0 whitelist gap and P0 buying-power zero-on-miss.
4. **State corruption**: API is stateless on our side; persistence is via portfolio_mgr (separate subsystem).
5. **Logic errors that pass tests**: place_order unified path likely lacks tests that assert account_id whitelist.
6. **Resource leaks**: AvanzaClient singleton (`get_instance()`); browser/Playwright lifecycle is in avanza_session, looks correct from grep.
7. **Time/timezone bugs**: `date.today()` is local timezone-naive — fine for Avanza which expects local Stockholm dates.
8. **API misuse**: stop-loss endpoint `/_api/trading/stoploss/new` correct (line 801). Trigger types and value_types valid.
9. **Trust boundary violations**: `f"/_api/trading/stoploss/{orderId}"` (metals_loop.py:4996) interpolates orderId into URL; orderId comes from Avanza's own response so trust boundary is internal, but if a stale state file had a manipulated orderId, you could DELETE arbitrary IDs. Minor risk.
10. **Incorrect partial-state assumptions**: P0 buying-power-zero, P1 expires_at-missing.
