"""List (and optionally cancel) Avanza open orders.

Usage:
    python scripts/avanza_orders.py            # list open orders
    python scripts/avanza_orders.py --cancel ORDER_ID  # cancel one order

Requires Avanza credentials or BankID session (see scripts/avanza_login.py).
"""

import argparse
from pprint import pprint

from portfolio.avanza_control import (
    delete_order,
    get_account_id,
    get_open_orders,
)


def list_orders() -> int:
    orders = get_open_orders()
    if not orders:
        print("No open orders.")
        return 0

    account = get_account_id()
    print(f"Open orders on account {account}:")
    for o in orders:
        # Avanza returns dict with keys: orderId, orderRequestStatus, state, price, volume, buy/sell
        order_id = o.get("orderId") or o.get("order_id")
        side = o.get("orderType") or o.get("order_type")
        state = o.get("state") or o.get("status")
        name = o.get("name") or o.get("orderBookName") or ""
        price = o.get("price")
        vol = o.get("volume")
        print(f"- {order_id} {side} {vol} @ {price} — {state} — {name}")
    return 0


def cancel_order(order_id: str) -> int:
    result = delete_order(order_id)
    print("Cancel result:")
    pprint(result)
    return 0


def main():
    parser = argparse.ArgumentParser(description="List or cancel Avanza open orders.")
    parser.add_argument("--cancel", metavar="ORDER_ID", help="Cancel a specific order id")
    args = parser.parse_args()

    if args.cancel:
        return cancel_order(args.cancel)
    return list_orders()


if __name__ == "__main__":
    raise SystemExit(main())
