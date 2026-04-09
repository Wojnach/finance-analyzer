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

            const makeResult = (acc) => ({
                buying_power: v(acc.buyingPower),
                total_value: v(acc.totalValue),
                own_capital: v(acc.ownCapital),
            });

            // Path A (legacy, pre-2026-04-09): data.categorizedAccounts
            const legacyCats = data.categorizedAccounts || [];
            for (const cat of legacyCats) {
                for (const acc of (cat.accounts || [])) {
                    if (acc && acc.accountId != null) ids_seen.push(String(acc.accountId));
                    if (String(acc.accountId) === accountId) return makeResult(acc);
                }
            }

            // Path B (new flat shape, 2026-04-09): data.accounts
            const flatAccounts = data.accounts || [];
            for (const acc of flatAccounts) {
                if (acc && acc.accountId != null) ids_seen.push(String(acc.accountId));
                if (String(acc.accountId) === accountId) return makeResult(acc);
            }

            // Path C (new categorized shape, 2026-04-09): data.categories
            const newCats = data.categories || [];
            for (const cat of newCats) {
                for (const acc of (cat.accounts || [])) {
                    if (acc && acc.accountId != null) ids_seen.push(String(acc.accountId));
                    if (String(acc.accountId) === accountId) return makeResult(acc);
                }
            }

            return {
                _error: 'no_account_match',
                legacy_category_count: legacyCats.length,
                flat_account_count: flatAccounts.length,
                new_category_count: newCats.length,
                ids_seen: ids_seen,
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
    """
    csrf = get_csrf(page)
    if not csrf:
        return False, {"error": "no CSRF token"}

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
