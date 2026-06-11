"""Delete failed order and retry sell for BULL GULD X20 AVA 6."""
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.avanza_session import (
    api_get,
    api_post,
    cancel_order,
    get_instrument_price,
    place_sell_order,
    verify_session,
)

ACCOUNT_ID = "1625505"
GOLD_OB_ID = "2308943"   # BULL GULD X20 AVA 6
VOLUME = 529

if not verify_session():
    print("ERROR: Session invalid.")
    sys.exit(1)

# Step 1: Delete all existing orders for this instrument
print("=== STEP 1: Delete existing orders ===")
orders_resp = api_get("/_api/trading/rest/orders")
orders_list = orders_resp.get("orders", orders_resp) if isinstance(orders_resp, dict) else orders_resp

for order in (orders_list if isinstance(orders_list, list) else []):
    oid = str(order.get("orderId", ""))
    order_ob = str(order.get("orderbookId", ""))
    state = order.get("state", "")
    side = order.get("side", "")
    print(f"  Order {oid}: ob={order_ob} side={side} state={state}")
    if order_ob == GOLD_OB_ID:
        print(f"    -> Deleting order {oid}...")
        del_result = cancel_order(oid, ACCOUNT_ID)
        print(f"    Delete result: {del_result}")

# Brief pause to let the deletion settle
time.sleep(2)

# Step 2: Get fresh price
print("\n=== STEP 2: Get fresh BULL GULD price ===")
price = get_instrument_price(GOLD_OB_ID)
if price:
    listing = price.get("listing", {})
    bid = listing.get("bidPrice")
    ask = listing.get("askPrice")
    last = listing.get("lastPrice")
    underlying_info = price.get("underlying", {})
    print(f"  Bid: {bid}")
    print(f"  Ask: {ask}")
    print(f"  Last: {last}")
    print(f"  Underlying: {underlying_info}")
else:
    print("  FAILED to get price")
    # Try raw API
    try:
        raw = api_get(f"/_api/market-guide/certificate/{GOLD_OB_ID}")
        print(f"  Raw: {json.dumps(raw, ensure_ascii=False)[:500]}")
    except Exception as e:
        print(f"  Raw fetch also failed: {e}")

# Step 3: Place sell order at bid
print("\n=== STEP 3: Place SELL order ===")
sell_price = None
if price:
    listing = price.get("listing", {})
    sell_price = listing.get("bidPrice") or listing.get("lastPrice")

if sell_price is None:
    print("  No price available, cannot place order")
    sys.exit(1)

print(f"  Selling {VOLUME} units at {sell_price} SEK")
print(f"  Total value: {VOLUME * sell_price:.0f} SEK")

result = place_sell_order(GOLD_OB_ID, sell_price, VOLUME, ACCOUNT_ID)
success = result.get("orderRequestStatus") == "SUCCESS"
print(f"\n  Success: {success}")
print(f"  Result: {json.dumps(result, indent=2, ensure_ascii=False)}")

if not success:
    # Retry with manual payload via api_post
    print("\n=== STEP 3b: Retry with manual payload ===")
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")

    result2 = api_post("/_api/trading-critical/rest/order/new", {
        "accountId": ACCOUNT_ID,
        "orderbookId": GOLD_OB_ID,
        "side": "SELL",
        "condition": "NORMAL",
        "price": sell_price,
        "validUntil": today_str,
        "volume": VOLUME,
    })
    print(f"  Result: {json.dumps(result2, indent=2, ensure_ascii=False)}")
