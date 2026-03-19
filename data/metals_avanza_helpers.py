"""Shared Avanza Playwright helpers for metals trading.

Canonical implementations of CSRF extraction, price fetching, order placement,
stop-loss placement, and session health checking. Used by both metals_loop.py
and metals_swing_trader.py to avoid code duplication.

All functions take an explicit Playwright `page` argument — they do NOT manage
their own browser instance.
"""

import datetime
import json


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
    except Exception:
        return None


def fetch_account_cash(page, account_id):
    """Fetch ISK buying power from Avanza accounts API.

    Returns dict with: buying_power, total_value, own_capital.
    Returns None on failure.
    """
    try:
        result = page.evaluate("""async (accountId) => {
            const resp = await fetch(
                'https://www.avanza.se/_api/account-overview/overview/categorizedAccounts',
                {credentials: 'include'}
            );
            if (resp.status !== 200) return null;
            const data = await resp.json();
            for (const cat of (data.categorizedAccounts || [])) {
                for (const acc of (cat.accounts || [])) {
                    if (String(acc.accountId) === accountId) {
                        const v = (x) => (x && typeof x === 'object' && 'value' in x) ? x.value : x;
                        return {
                            buying_power: v(acc.buyingPower),
                            total_value: v(acc.totalValue),
                            own_capital: v(acc.ownCapital),
                        };
                    }
                }
            }
            return null;
        }""", account_id)
        return result
    except Exception:
        return None


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
    except Exception:
        return False, ""


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
    except Exception:
        return False
