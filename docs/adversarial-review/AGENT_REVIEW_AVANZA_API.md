# Agent Adversarial Review: avanza-api

**Agent**: feature-dev:code-reviewer
**Subsystem**: avanza-api (2,298 lines, 5 files)
**Duration**: ~198 seconds
**Findings**: 13 (2 P0, 7 P1, 3 P2, 1 P3)

---

## P0 Findings (CRITICAL — Money-Losing)

### A-AV-1: Playwright Context Used Outside Lock — Concurrent Request Race [P0]
- **File**: `portfolio/avanza_session.py:183-207, 219-261, 263-291`
- **Description**: `_get_playwright_context()` acquires `_pw_lock` only during context *creation*. After returning the context, all callers (`api_get`, `api_post`, `api_delete`) use `ctx.request.*` WITHOUT holding the lock. Playwright's sync_api is NOT thread-safe — concurrent use from the main loop's ThreadPoolExecutor (8 workers) and metals loop's 10s fast-tick thread will corrupt internal state.
- **Impact**: Corrupt trade responses — a BUY confirmation could be consumed by a different request. Order failures, double-executions, or wrong-asset trades.
- **Fix**: Hold `_pw_lock` for the entire duration of each `api_get/api_post/api_delete` call.

### A-AV-2: TOTP Path Has No Account Whitelist — Pension Account Can Receive Orders [P0]
- **File**: `portfolio/avanza_client.py:296-320`
- **Description**: `avanza_client._place_order()` uses `get_account_id()` which discovers the ISK account by scanning for "ISK" in accountType. There is NO hardcoded allowlist and NO comparison against "1625505". If Avanza re-orders accounts in the API response, pension account 2674244 could receive trades.
- **Impact**: Real money trades on pension account. Different tax treatment, withdrawal restrictions, regulatory risk.
- **Fix**: Assert `_account_id == "1625505"` in `get_account_id()` before caching.

---

## P1 Findings

### A-AV-3: get_positions() and get_portfolio_value() Include Pension Account [P1]
- **File**: `portfolio/avanza_client.py:152-200`
- **Description**: Both iterate over ALL accounts without filtering by account ID. Pension account positions and value are included.
- **Impact**: Inflated portfolio value, incorrect position sizing, drawdown calculations corrupted.

### A-AV-4: Single CONFIRM Matches Most-Recent Order — Ambiguous Confirmation [P1]
- **File**: `portfolio/avanza_orders.py:111-143`
- **Description**: When multiple pending orders exist, bare "CONFIRM" matches the most recent one. Earlier pending orders silently age out. No disambiguation for the user.
- **Impact**: Wrong order confirmed. Capital deployed to unintended positions.

### A-AV-5: _execute_confirmed_order Uses TOTP Path Not BankID [P1]
- **File**: `portfolio/avanza_orders.py:17, 213-224`
- **Description**: Imports `place_buy_order/place_sell_order` from `avanza_control` which re-exports from `avanza_client` (TOTP library path), NOT from `avanza_session` (BankID path).
- **Impact**: Orders fail silently if TOTP client not initialized (common when BankID is primary).

### A-AV-6: get_quote() Hardcodes "stock" Type [P1]
- **File**: `portfolio/avanza_session.py:562-568`
- **Description**: Always hits `/_api/market-guide/stock/{id}/quote`. Returns wrong data for certificates/warrants.
- **Impact**: Wrong price for 5x leverage warrants. 2% price error = 10% position error.

### A-AV-7: _try_session_auth Never Re-checks Expiry [P1]
- **File**: `portfolio/avanza_client.py:55-69`
- **Description**: Once `verify_session()` returns True, `_session_client = True` is cached permanently. Session expiry at 24h causes first subsequent request to fail with 401.
- **Impact**: First request after session expiry fails. Recovery works but with one failed API call.

### A-AV-8: stopLossOrderEvent.priceType Blindly Inherits value_type [P1]
- **File**: `portfolio/avanza_session.py:661-667`
- **Description**: `priceType` is set to `value_type`, which for trailing stops is "PERCENTAGE". If any caller passes `value_type="PERCENTAGE"` for a non-trailing stop, the sell price is interpreted as a percentage.
- **Impact**: Stop-loss placed with wrong price interpretation.

### A-AV-9: Pending Orders TOCTOU Race — Potential Double Execution [P1]
- **File**: `portfolio/avanza_orders.py:111-143`
- **Description**: `check_pending_orders()` reads, mutates, then writes the pending list with no locking. Concurrent calls could both load the same order and execute it twice.
- **Impact**: Order executed twice, doubling position size unintentionally.

---

## P2 Findings

### A-AV-10: Session Cookies Stored as Plaintext in Repo Root [P2]
- **File**: `portfolio/avanza_session.py:23-25`
- **Description**: BankID session cookies at `data/avanza_session.json` and `data/avanza_storage_state.json` are plaintext in the repo root. Not git-ignored.

### A-AV-11: reset_client() Does Not Clear Cached _account_id [P2]
- **File**: `portfolio/avanza_client.py:101-104`
- **Description**: After re-auth, stale cached account ID is still returned.

### A-AV-12: Order Status Set Before Save — Crash Window [P2]
- **File**: `portfolio/avanza_orders.py:132-142`
- **Description**: `order["status"] = "confirmed"` set before `_save_pending()`. Crash between these lines leaves order in stale state.

---

## P3 Findings

### A-AV-13: api_get(**kwargs) Silently Drops All Keyword Arguments [P3]
- **File**: `portfolio/avanza_session.py:184`
