"""Delete existing orders then sell BULL GULD X20 AVA 6."""
import sys, json, time
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
GOLD_OB_ID = "2308943"
VOLUME = 529

if not verify_session():
    print("ERROR: Session invalid.")
    sys.exit(1)

# Step 1: Get all orders and delete ones for gold bull
print("=== STEP 1: Delete existing gold orders ===")
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
        print(f"    Result: {del_result}")

print("\nWaiting 3s for order deletion to settle...")
time.sleep(3)

# Step 2: Verify orders are gone
print("\n=== STEP 2: Verify orders deleted ===")
orders2 = api_get("/_api/trading/rest/orders")
orders2_list = orders2.get("orders", orders2) if isinstance(orders2, dict) else orders2
gold_orders = [o for o in (orders2_list if isinstance(orders2_list, list) else []) if str(o.get("orderbookId","")) == GOLD_OB_ID]
print(f"  Remaining gold orders: {len(gold_orders)}")
for o in gold_orders:
    print(f"    {o.get('orderId')} state={o.get('state')} side={o.get('side')}")

# Step 3: Get fresh price
print("\n=== STEP 3: Get fresh price ===")
price = get_instrument_price(GOLD_OB_ID)
if price:
    listing = price.get("listing", {})
    bid = listing.get("bidPrice")
    ask = listing.get("askPrice")
    last = listing.get("lastPrice")
    print(f"  Bid: {bid}, Ask: {ask}, Last: {last}")
else:
    print("  FAILED to get price!")
    sys.exit(1)

# Step 4: Place sell at bid
sell_price = bid or last
print(f"\n=== STEP 4: Place SELL {VOLUME} @ {sell_price} SEK ===")
print(f"  Total: {VOLUME * sell_price:.0f} SEK")

result = place_sell_order(GOLD_OB_ID, sell_price, VOLUME, ACCOUNT_ID)
print(f"\n  Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
