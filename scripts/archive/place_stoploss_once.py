"""Place cascading stop-loss orders for silver and gold warrants.

One-time script. Places via /_api/trading/stoploss/new (the CORRECT stop-loss API).
NEVER use /_api/trading-critical/rest/order/new for stop-losses.

Usage: .venv/Scripts/python.exe data/place_stoploss_once.py [--dry-run]
"""
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.avanza_session import api_delete, api_post, verify_session

DRY_RUN = "--dry-run" in sys.argv

# ── Stop-loss IDs to delete first ──────────────────────────────
DELETE_IDS = [
    "A2^1773297348702^211738",  # old silver single stop
    "A2^1773297348702^211742",  # old gold single stop
]

# ── Cascading stop-loss definitions ────────────────────────────
# 1% above knock-out barrier, split into 3 tighter levels each.
ORDERS = [
    # --- SILVER: 3 cascading levels, 4031 units total ---
    {
        "label": "SILVER L1 (slow decline)",
        "account_id": "1625505",
        "orderbook_id": "2334960",
        "trigger_price": 2.30,
        "sell_price": 2.15,
        "volume": 1344,
        "valid_days": 8,
        "current_bid": 5.09,
    },
    {
        "label": "SILVER L2 (continued drop)",
        "account_id": "1625505",
        "orderbook_id": "2334960",
        "trigger_price": 2.15,
        "sell_price": 2.00,
        "volume": 1344,
        "valid_days": 8,
        "current_bid": 5.09,
    },
    {
        "label": "SILVER L3 (last resort)",
        "account_id": "1625505",
        "orderbook_id": "2334960",
        "trigger_price": 2.00,
        "sell_price": 1.70,
        "volume": 1343,
        "valid_days": 8,
        "current_bid": 5.09,
    },
    # --- GOLD: 3 cascading levels, 424 units total ---
    {
        "label": "GOLD L1 (slow decline)",
        "account_id": "1625505",
        "orderbook_id": "2319821",
        "trigger_price": 18.00,
        "sell_price": 16.50,
        "volume": 142,
        "valid_days": 8,
        "current_bid": 30.00,
    },
    {
        "label": "GOLD L2 (continued drop)",
        "account_id": "1625505",
        "orderbook_id": "2319821",
        "trigger_price": 16.50,
        "sell_price": 15.00,
        "volume": 141,
        "valid_days": 8,
        "current_bid": 30.00,
    },
    {
        "label": "GOLD L3 (last resort)",
        "account_id": "1625505",
        "orderbook_id": "2319821",
        "trigger_price": 15.00,
        "sell_price": 13.00,
        "volume": 141,
        "valid_days": 8,
        "current_bid": 30.00,
    },
]


def safety_checks(order: dict) -> list[str]:
    errors = []
    trigger = order["trigger_price"]
    sell = order["sell_price"]
    bid = order["current_bid"]

    if trigger >= bid:
        errors.append(f"DANGER: trigger {trigger} >= bid {bid} -- instant sell!")
    if (bid - trigger) / bid * 100 < 3.0:
        errors.append(f"Too close: trigger {trigger} is <3% below bid {bid}")
    if sell >= trigger:
        errors.append(f"Sell {sell} >= trigger {trigger}")
    if order["volume"] <= 0:
        errors.append("Volume must be positive")
    if sell <= 0:
        errors.append("Sell price must be positive")
    return errors


def delete_stoploss(stop_id: str) -> dict:
    if DRY_RUN:
        return {"dry_run": True}

    return api_delete(f"/_api/trading/stoploss/{stop_id}")


def place_one_stoploss(order: dict) -> dict:
    valid_until = (
        datetime.datetime.now() + datetime.timedelta(days=order["valid_days"])
    ).strftime("%Y-%m-%d")

    payload = {
        "parentStopLossId": "0",
        "accountId": order["account_id"],
        "orderBookId": order["orderbook_id"],
        "stopLossTrigger": {
            "type": "LESS_OR_EQUAL",
            "value": order["trigger_price"],
            "validUntil": valid_until,
            "valueType": "MONETARY",
            "triggerOnMarketMakerQuote": True,
        },
        "stopLossOrderEvent": {
            "type": "SELL",
            "price": order["sell_price"],
            "volume": order["volume"],
            "validDays": order["valid_days"],
            "priceType": "MONETARY",
            "shortSellingAllowed": False,
        },
    }

    if DRY_RUN:
        return {"dry_run": True, "payload": payload}

    body = api_post("/_api/trading/stoploss/new", payload)

    return {
        "body": body,
        "success": body.get("status") == "SUCCESS" if isinstance(body, dict) else False,
        "stop_id": body.get("stoplossOrderId", "") if isinstance(body, dict) else "",
    }


def main():
    mode = "DRY RUN" if DRY_RUN else "LIVE"
    print(f"{'=' * 60}")
    print(f"CASCADING STOP-LOSS PLACEMENT  [{mode}]")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    print("\n[1] Verifying Avanza session...")
    if not verify_session():
        print("ERROR: Session invalid.")
        sys.exit(1)
    print("  Session OK")

    # Step 1: Delete old single stop-losses
    print("\n[2] Deleting old stop-losses...")
    for sid in DELETE_IDS:
        r = delete_stoploss(sid)
        status = "DRY RUN" if r.get("dry_run") else ("OK" if r.get("ok") else f"FAIL {r}")
        print(f"  {sid}: {status}")

    # Step 2: Place cascading stops
    print("\n[3] Placing 6 cascading stop-losses...")
    results = []
    for i, order in enumerate(ORDERS, 1):
        label = order["label"]
        errors = safety_checks(order)
        if errors:
            print(f"\n  [{i}] {label} -- SAFETY FAIL: {errors}")
            results.append({"label": label, "error": errors})
            continue

        result = place_one_stoploss(order)
        results.append({"label": label, **result})

        if DRY_RUN:
            print(f"  [{i}] {label}: trigger {order['trigger_price']} sell {order['sell_price']} x{order['volume']} -- DRY RUN")
        elif result.get("success"):
            print(f"  [{i}] {label}: trigger {order['trigger_price']} sell {order['sell_price']} x{order['volume']} -- OK (ID: {result['stop_id']})")
        else:
            print(f"  [{i}] {label}: FAILED -- {result.get('body', result.get('error', '?'))}")

    # Summary
    placed = sum(1 for r in results if r.get("success"))
    failed = sum(1 for r in results if not r.get("success") and not r.get("dry_run"))
    print(f"\n{'=' * 60}")
    print(f"DONE: {placed} placed, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
