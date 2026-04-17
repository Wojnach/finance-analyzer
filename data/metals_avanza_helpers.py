"""Shared Avanza Playwright helpers for metals trading.

Canonical implementations of CSRF extraction, price fetching, order placement,
stop-loss placement, and session health checking. Used by both metals_loop.py
and metals_swing_trader.py to avoid code duplication.

All functions take an explicit Playwright `page` argument — they do NOT manage
their own browser instance.
"""

import datetime
import json
import logging

from portfolio.avanza_order_lock import avanza_order_lock

logger = logging.getLogger("metals_avanza_helpers")


def get_csrf(page):
    """Extract CSRF token from Avanza cookies."""
    for c in page.context.cookies():
        if c["name"] == "AZACSRF":
            return c["value"]
    return None


def fetch_price(page, ob_id, api_type):
    """Fetch live price from Avanza market-guide API.

    Returns dict with: bid, ask, last, change_pct, high, low,
    underlying, underlying_name, leverage, barrier.
    Returns None on failure.
    """
    try:
        result = page.evaluate("""async (args) => {
            const [id, type] = args;
            const resp = await fetch('https://www.avanza.se/_api/market-guide/' + type + '/' + id,
                {credentials:'include'});
            if (resp.status !== 200) return null;
            const d = await resp.json();
            const v = (x) => (x && typeof x === 'object' && 'value' in x) ? x.value : x;
            return {
                bid: v(d.quote?.buy), ask: v(d.quote?.sell), last: v(d.quote?.last),
                change_pct: v(d.quote?.changePercent),
                high: v(d.quote?.highest), low: v(d.quote?.lowest),
                underlying: v(d.underlying?.quote?.last),
                underlying_name: d.underlying?.name,
                leverage: v(d.keyIndicators?.leverage),
                barrier: v(d.keyIndicators?.barrierLevel),
            };
        }""", [ob_id, api_type])
        return result
    except Exception as e:
        logger.warning(
            "fetch_price: exception=%r", e, exc_info=True,
        )
        return None


def fetch_positions(page, account_id):
    """Fetch current positions for an Avanza account, keyed by orderbook id.

    Used by the swing trader to reconcile its internal position state against
    what Avanza actually holds. Returns `None` on transient failure so the
    caller can distinguish "session down" from "legitimately empty account"
    (which returns `{}`).

    Returns:
        dict[str, dict] | None — map of orderbook_id → {name, units, value,
        avg_price, api_type}. Empty dict means the account has no positions.
        None means the API call failed and the caller should retry later.
    """
    try:
        result = page.evaluate("""async (accountId) => {
            const resp = await fetch(
                'https://www.avanza.se/_api/position-data/positions',
                {credentials: 'include'}
            );
            if (resp.status !== 200) return null;
            const data = await resp.json();
            const v = (x) => (x && typeof x === 'object' && 'value' in x) ? x.value : x;
            const out = {};
            for (const entry of (data.withOrderbook || [])) {
                const inst = entry.instrument || {};
                const ob = inst.orderbook || {};
                const acc = entry.account || {};
                if (accountId && String(acc.id || '') !== accountId) continue;
                const obId = String(ob.id || '');
                if (!obId) continue;
                out[obId] = {
                    name: inst.name || ob.name || '',
                    units: v(entry.volume) || 0,
                    value: v(entry.value) || 0,
                    avg_price: v(entry.averageAcquiredPrice) || 0,
                    api_type: (inst.type || '').toLowerCase(),
                };
            }
            return out;
        }""", str(account_id) if account_id else "")
        return result if isinstance(result, dict) else None
    except Exception as e:
        logger.warning(
            "fetch_positions: exception=%r", e, exc_info=True,
        )
        return None


def fetch_account_cash(page, account_id):
    """Fetch ISK buying power from Avanza accounts API.

    Returns dict with: buying_power, total_value, own_capital on success.
    Returns None on failure. Failure modes are logged via the module logger
    with diagnostic context — the JS layer returns a `_error` dict on
    iteration failure so we can distinguish "HTTP error" from "account ID
    not found" from "Avanza renamed a field" without guessing.

    2026-04-09 afternoon (Fix 4b): Avanza changed the response shape. The
    field used to be `data.categorizedAccounts` (array of category objects,
    each with an `accounts` array). The new shape exposes three top-level
    keys — `categories`, `accounts`, `loans` — where `accounts` is a flat
    array of all user accounts. Diagnostic run at 18:41:11 CET confirmed:
    `top_level_keys: ['categories', 'accounts', 'loans']`. We now try the
    legacy categorized path first (forward-compat), then the new flat
    `data.accounts` path, then the new `data.categories` path, taking
    whichever finds the target account. Diagnostic on total miss includes
    both the categorized count and the flat count so the next regression
    is equally easy to spot.
    """
    try:
        result = page.evaluate("""async (accountId) => {
            const resp = await fetch(
                'https://www.avanza.se/_api/account-overview/overview/categorizedAccounts',
                {credentials: 'include'}
            );
            if (resp.status !== 200) {
                return {_error: 'http', status: resp.status};
            }
            const data = await resp.json();
            const v = (x) => (x && typeof x === 'object' && 'value' in x) ? x.value : x;
            const ids_seen = [];
            let sample_account_keys = null;

            // 2026-04-09 Fix 4c: Avanza changed the per-account id field.
            // The old shape used `acc.accountId`; the new shape uses `acc.id`
            // (other Avanza endpoints like position-data/positions already
            // use `acc.id` — see fetch_positions at line ~86 in this file).
            // Try every known ID field in order; on total miss, return the
            // actual key list of the first account we saw so the next
            // diagnostic cycle immediately reveals any further renames.
            const getAccId = (acc) => {
                if (!acc) return null;
                return acc.accountId
                    || acc.id
                    || acc.accountNumber
                    || acc.number
                    || null;
            };

            // Same treatment for the balance field — we haven't confirmed
            // yet that `buyingPower` survived the 2026-04-09 shape change,
            // so try common alternates if the primary field is missing.
            // `v()` unwraps {value: N} → N when Avanza wraps numeric fields.
            const getBalance = (acc, primary, alternates) => {
                if (acc == null) return undefined;
                const p = v(acc[primary]);
                if (p != null) return p;
                for (const alt of alternates) {
                    const x = v(acc[alt]);
                    if (x != null) return x;
                }
                return undefined;
            };

            const makeResult = (acc) => ({
                buying_power: getBalance(acc, 'buyingPower',
                    ['buyingPowerAvailable', 'availableCash', 'availableFunds']),
                total_value: getBalance(acc, 'totalValue',
                    ['accountTotalValue', 'totalHoldings']),
                own_capital: getBalance(acc, 'ownCapital',
                    ['netDeposit', 'selfOwnedCapital']),
            });

            const checkAccount = (acc) => {
                if (sample_account_keys === null && acc && typeof acc === 'object') {
                    sample_account_keys = Object.keys(acc);
                }
                const id = getAccId(acc);
                if (id != null) ids_seen.push(String(id));
                if (String(id) === accountId) return makeResult(acc);
                return null;
            };

            // Path A (legacy, pre-2026-04-09): data.categorizedAccounts
            const legacyCats = data.categorizedAccounts || [];
            for (const cat of legacyCats) {
                for (const acc of (cat.accounts || [])) {
                    const r = checkAccount(acc);
                    if (r) return r;
                }
            }

            // Path B (new flat shape, 2026-04-09): data.accounts
            const flatAccounts = data.accounts || [];
            for (const acc of flatAccounts) {
                const r = checkAccount(acc);
                if (r) return r;
            }

            // Path C (new categorized shape, 2026-04-09): data.categories
            const newCats = data.categories || [];
            for (const cat of newCats) {
                for (const acc of (cat.accounts || [])) {
                    const r = checkAccount(acc);
                    if (r) return r;
                }
            }

            return {
                _error: 'no_account_match',
                legacy_category_count: legacyCats.length,
                flat_account_count: flatAccounts.length,
                new_category_count: newCats.length,
                ids_seen: ids_seen,
                sample_account_keys: sample_account_keys,
                top_level_keys: Object.keys(data),
            };
        }""", account_id)
    except Exception as e:
        logger.warning(
            "fetch_account_cash: page.evaluate raised %r",
            e, exc_info=True,
        )
        return None

    if result is None:
        logger.warning(
            "fetch_account_cash: page.evaluate returned None "
            "(navigation issue or Playwright session drift?)"
        )
        return None

    if isinstance(result, dict) and result.get("_error"):
        logger.warning(
            "fetch_account_cash: diagnostic failure account_id=%s result=%s",
            account_id, result,
        )
        return None

    return result  # success path — {buying_power, total_value, own_capital}


def place_order(page, account_id, ob_id, side, price, volume):
    """Place a BUY or SELL order on Avanza.

    Returns (success: bool, result: dict).
    Result dict contains: http_status, parsed (response body), order_id.

    Returns (False, {"error": ...}) only for hard refusals like a
    missing CSRF token. Orders below the 1000 SEK courtage threshold
    are logged as WARNINGs but NOT refused — metals swing-trader
    callers floor budget-to-whole-units (`units = int(alloc / ask)`),
    which can legitimately produce sub-1000 SEK BUYs on expensive
    warrants even when the allocation was exactly 1000 SEK. Refusing
    would strand floor-sized entries (codex P2 2026-04-17).
    """
    csrf = get_csrf(page)
    if not csrf:
        return False, {"error": "no CSRF token"}

    # 2026-04-17: 1000 SEK min-courtage threshold — warn but proceed.
    try:
        total = round(float(price) * float(volume), 2)
    except (TypeError, ValueError):
        total = 0.0
    if 0 < total < 1000.0:
        logger.warning(
            "place_order: %s total %.2f SEK below 1000 SEK courtage threshold "
            "(ob=%s vol=%s price=%s) — proceeding (fees elevated)",
            side, total, ob_id, volume, price,
        )

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    payload = {
        "accountId": account_id,
        "orderbookId": ob_id,
        "side": side,
        "condition": "NORMAL",
        "price": price,
        "validUntil": today_str,
        "volume": volume,
    }

    try:
        # 2026-04-13: cross-process order lock — prevents metals_loop,
        # golddigger, main.py (via avanza_session) from racing on buying_power.
        # Busy peer: OrderLockBusyError bubbles up (caller retries next cycle).
        with avanza_order_lock(op=f"place_order/{side}/{ob_id}"):
            result = page.evaluate("""async (args) => {
                const [payload, token] = args;
                const resp = await fetch('https://www.avanza.se/_api/trading-critical/rest/order/new', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-SecurityToken': token},
                    credentials: 'include',
                    body: JSON.stringify(payload),
                });
                return {status: resp.status, body: await resp.text()};
            }""", [payload, csrf])

        body = {}
        try:
            body = json.loads(result.get("body", ""))
        except (json.JSONDecodeError, TypeError):
            pass

        success = body.get("orderRequestStatus") == "SUCCESS"
        order_id = body.get("orderId", "")
        return success, {
            "http_status": result.get("status"),
            "parsed": body,
            "order_id": order_id,
        }
    except Exception as e:
        logger.warning(
            "place_order: exception=%r", e, exc_info=True,
        )
        return False, {"error": str(e)}


def place_stop_loss(page, account_id, ob_id, trigger_price, sell_price, volume,
                    valid_days=8):
    """Place a hardware stop-loss via the Avanza stop-loss API.

    IMPORTANT: Uses /_api/trading/stoploss/new, NOT the regular order API.
    The regular order API causes "crossing prices" errors for stop-losses.

    Returns (success: bool, stop_id: str).
    """
    csrf = get_csrf(page)
    if not csrf:
        return False, ""

    # 2026-04-17: warn (don't refuse) on sub-1000 SEK stop legs —
    # metals_loop cascades stops into ≤3 legs and per-leg value can
    # legitimately fall below the courtage threshold. Surfacing via
    # log lets callers audit fee impact without breaking cascading.
    try:
        leg_total = round(float(sell_price) * float(volume), 2)
    except (TypeError, ValueError):
        leg_total = 0.0
    if 0 < leg_total < 1000.0:
        logger.warning(
            "place_stop_loss leg %.2f SEK below 1000 SEK courtage threshold "
            "(vol=%s sell=%s ob=%s)",
            leg_total, volume, sell_price, ob_id,
        )

    valid_until = (datetime.datetime.now()
                   + datetime.timedelta(days=valid_days)).strftime("%Y-%m-%d")
    payload = {
        "parentStopLossId": "0",
        "accountId": account_id,
        "orderBookId": ob_id,
        "stopLossTrigger": {
            "type": "LESS_OR_EQUAL",
            "value": trigger_price,
            "validUntil": valid_until,
            "valueType": "MONETARY",
            "triggerOnMarketMakerQuote": True,
        },
        "stopLossOrderEvent": {
            "type": "SELL",
            "price": sell_price,
            "volume": volume,
            "validDays": valid_days,
            "priceType": "MONETARY",
            "shortSellingAllowed": False,
        },
    }

    try:
        # 2026-04-13: cross-process order lock (see place_order rationale).
        with avanza_order_lock(op=f"place_stop_loss/{ob_id}"):
            result = page.evaluate("""async (args) => {
                const [payload, token] = args;
                const resp = await fetch('https://www.avanza.se/_api/trading/stoploss/new', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-SecurityToken': token},
                    credentials: 'include',
                    body: JSON.stringify(payload),
                });
                return {status: resp.status, body: await resp.text()};
            }""", [payload, csrf])

        body = {}
        try:
            body = json.loads(result.get("body", ""))
        except (json.JSONDecodeError, TypeError):
            pass

        success = body.get("status") == "SUCCESS"
        stop_id = body.get("stoplossOrderId", "")
        return success, stop_id
    except Exception as e:
        logger.warning(
            "place_stop_loss: exception=%r", e, exc_info=True,
        )
        return False, ""


def delete_order(page, account_id, order_id):
    """Cancel an open order on Avanza.

    IMPORTANT: Uses POST to /_api/trading-critical/rest/order/delete, NOT
    the DELETE HTTP method. The DELETE verb returns 404 on this endpoint
    (discovered 2026-03-24 — Avanza changed the API at some point).

    Returns (success: bool, result: dict).
    """
    csrf = get_csrf(page)
    if not csrf:
        return False, {"error": "no CSRF token"}

    try:
        # 2026-04-13: cross-process order lock (see place_order rationale).
        with avanza_order_lock(op=f"delete_order/{order_id}"):
            result = page.evaluate("""async (args) => {
                const [accountId, orderId, token] = args;
                const resp = await fetch('https://www.avanza.se/_api/trading-critical/rest/order/delete', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-SecurityToken': token},
                    credentials: 'include',
                    body: JSON.stringify({accountId: accountId, orderId: orderId}),
                });
                return {status: resp.status, body: await resp.text()};
            }""", [account_id, order_id, csrf])

        body = {}
        try:
            body = json.loads(result.get("body", ""))
        except (json.JSONDecodeError, TypeError):
            pass

        success = body.get("orderRequestStatus") == "SUCCESS"
        return success, {
            "http_status": result.get("status"),
            "parsed": body,
            "order_id": order_id,
        }
    except Exception as e:
        logger.warning(
            "delete_order: exception=%r", e, exc_info=True,
        )
        return False, {"error": str(e)}


def delete_stop_loss(page, account_id, stop_id):
    """Delete a stop-loss order on Avanza.

    Uses DELETE to /_api/trading/stoploss/{accountId}/{stopId}.

    Returns (success: bool, result: dict).
    """
    csrf = get_csrf(page)
    if not csrf:
        return False, {"error": "no CSRF token"}

    try:
        result = page.evaluate("""async (args) => {
            const [accountId, stopId, token] = args;
            const resp = await fetch(
                'https://www.avanza.se/_api/trading/stoploss/' + accountId + '/' + stopId,
                {
                    method: 'DELETE',
                    headers: {'Content-Type': 'application/json', 'X-SecurityToken': token},
                    credentials: 'include',
                }
            );
            return {status: resp.status, body: await resp.text()};
        }""", [account_id, stop_id, csrf])

        http_status = result.get("status", 0)
        success = 200 <= http_status < 300
        return success, {"http_status": http_status}
    except Exception as e:
        logger.warning(
            "delete_stop_loss: exception=%r", e, exc_info=True,
        )
        return False, {"error": str(e)}


def check_session_alive(page):
    """Quick 401 check — returns True if Avanza session is alive."""
    try:
        result = page.evaluate("""async () => {
            const resp = await fetch(
                'https://www.avanza.se/_api/account-overview/overview/categorizedAccounts',
                {credentials: 'include'}
            );
            return resp.status;
        }""")
        return result == 200
    except Exception as e:
        logger.warning(
            "check_session_alive: exception=%r", e, exc_info=True,
        )
        return False
