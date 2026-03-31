"""Debug gold sell: verify position, check IDs, retry order."""
import sys, json, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.avanza_session import (
    api_get,
    api_post,
    cancel_order,
    get_positions,
    verify_session,
    _get_csrf,
)

ACCOUNT_ID = "1625505"
GOLD_OB_ID = "2308943"

if not verify_session():
    print("ERROR: Session invalid.")
    sys.exit(1)

# Step 1: Get position data to verify holdings
print("=== STEP 1: Verify position via positions API ===")
pos_data = api_get("/_api/position-data/positions")

# Find gold bull position
gold_pos = None
for item in pos_data.get("withOrderbook", []):
    ob = item.get("instrument", {}).get("orderbook", {})
    ob_id = str(ob.get("id", ""))
    name = ob.get("name", "")
    acct = str(item.get("account", {}).get("id", ""))
    vol = item.get("volume", {}).get("value", 0) if item.get("volume") else 0
    if "GULD" in name.upper() or "GOLD" in name.upper() or ob_id == GOLD_OB_ID:
        print(f"  FOUND: {name}")
        print(f"    orderbook_id: {ob_id}")
        print(f"    instrument_type: {ob.get('type', 'unknown')}")
        print(f"    account_id: {acct}")
        print(f"    volume: {vol}")
        print(f"    full_instrument: {json.dumps(item.get('instrument', {}), indent=2)[:500]}")
        print(f"    full_account: {json.dumps(item.get('account', {}), indent=2)[:300]}")
        gold_pos = {"ob_id": ob_id, "acct": acct, "vol": int(vol), "name": name}

if not gold_pos:
    print("  NO GOLD POSITION FOUND! Checking all positions:")
    for item in pos_data.get("withOrderbook", []):
        ob = item.get("instrument", {}).get("orderbook", {})
        name = ob.get("name", "")
        ob_id = str(ob.get("id", ""))
        acct = str(item.get("account", {}).get("id", ""))
        vol = item.get("volume", {}).get("value", 0) if item.get("volume") else 0
        print(f"  - {name} | ob:{ob_id} acct:{acct} vol:{vol}")
    sys.exit(1)

# Step 2: Check existing orders
print("\n=== STEP 2: Check existing orders ===")
orders = api_get("/_api/trading/rest/orders")
print(f"  Orders response: {json.dumps(orders, indent=2)[:1000]}")

# Step 3: Get current bid price
print(f"\n=== STEP 3: Get current price for {gold_pos['ob_id']} ===")
price_data = api_get(f"/_api/market-guide/certificate/{gold_pos['ob_id']}")
listing = price_data.get("listing", {}) if isinstance(price_data, dict) else {}
price_summary = {
    "lastPrice": listing.get("lastPrice"),
    "bidPrice": listing.get("bidPrice"),
    "askPrice": listing.get("askPrice"),
    "change": listing.get("change"),
    "changePct": listing.get("changePercent"),
    "updated": listing.get("updated"),
    "tradable": listing.get("tradable"),
    "buyable": listing.get("buyable"),
    "sellable": listing.get("sellable"),
    "shortSellable": listing.get("shortSellable"),
}
print(f"  Price data: {json.dumps(price_summary, indent=2)}")

# Step 4: Delete existing orders for this instrument
print("\n=== STEP 4: Delete existing orders ===")
orders_list = orders if isinstance(orders, list) else orders.get("orders", [])
for order in orders_list:
    oid = order.get("orderId") or order.get("id")
    order_ob = str(order.get("orderbookId", ""))
    if order_ob == gold_pos["ob_id"]:
        print(f"  Deleting order {oid} for {order_ob}")
        del_result = cancel_order(str(oid), gold_pos["acct"])
        print(f"    Delete result: {del_result}")

# Step 5: Place sell order using VERIFIED ob_id and account
print(f"\n=== STEP 5: Place SELL order ===")
print(f"  Using: ob_id={gold_pos['ob_id']}, acct={gold_pos['acct']}, vol={gold_pos['vol']}")

sell_price = price_summary.get("bidPrice") or price_summary.get("lastPrice")
print(f"  Sell price: {sell_price}")

today_str = datetime.datetime.now().strftime("%Y-%m-%d")
payload = {
    "accountId": gold_pos["acct"],
    "orderbookId": gold_pos["ob_id"],
    "side": "SELL",
    "condition": "NORMAL",
    "price": sell_price,
    "validUntil": today_str,
    "volume": gold_pos["vol"],
}
print(f"  Payload: {json.dumps(payload)}")

result = api_post("/_api/trading-critical/rest/order/new", payload)

print(f"\n=== RESULT ===")
print(f"  Result: {json.dumps(result, indent=2)}")
