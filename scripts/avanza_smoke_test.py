"""Live smoke test for the new Avanza package.

Run: .venv/Scripts/python.exe scripts/avanza_smoke_test.py
Requires valid TOTP credentials in config.json.
"""
import json, sys, time
sys.path.insert(0, ".")

from portfolio.file_utils import load_json

config = load_json("config.json")
if not config or "avanza" not in config:
    print("ERROR: config.json missing or no avanza section")
    sys.exit(1)

from portfolio.avanza.client import AvanzaClient

print("1. Authenticating with TOTP...")
t0 = time.perf_counter()
client = AvanzaClient.get_instance(config)
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — authenticated in {dt:.0f}ms")
print(f"   Push subscription ID: {client.push_subscription_id[:12]}...")
print()

from portfolio.avanza.account import get_positions, get_buying_power
print("2. Fetching positions...")
t0 = time.perf_counter()
positions = get_positions(account_id="1625505")
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — {len(positions)} positions in {dt:.0f}ms")
for p in positions[:5]:
    print(f"   {p.name}: {p.volume}x @ {p.last_price} SEK (P&L: {p.profit:+.0f})")
print()

print("3. Fetching buying power...")
t0 = time.perf_counter()
cash = get_buying_power()
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — BP: {cash.buying_power:,.0f} SEK, Total: {cash.total_value:,.0f} ({dt:.0f}ms)")
print()

from portfolio.avanza.market_data import get_quote, get_market_data
print("4. Fetching quote (856394 = BULL GULD X8)...")
t0 = time.perf_counter()
try:
    q = get_quote("856394", instrument_type="certificate")
    dt = (time.perf_counter() - t0) * 1000
    print(f"   OK — bid={q.bid} ask={q.ask} last={q.last} ({dt:.0f}ms)")
except Exception as e:
    print(f"   WARN — quote failed: {e}")
print()

print("5. Fetching market data + order depth...")
t0 = time.perf_counter()
try:
    md = get_market_data("856394")
    dt = (time.perf_counter() - t0) * 1000
    print(f"   OK — {len(md.bid_levels)} bid, {len(md.ask_levels)} ask levels ({dt:.0f}ms)")
except Exception as e:
    print(f"   WARN — market data failed: {e}")
print()

from portfolio.avanza.search import search
print("6. Searching instruments...")
t0 = time.perf_counter()
try:
    results = search("MINI L SILVER", limit=3)
    dt = (time.perf_counter() - t0) * 1000
    print(f"   OK — {len(results)} results ({dt:.0f}ms)")
    for r in results:
        print(f"   {r.orderbook_id} | {r.name} | {r.instrument_type}")
except Exception as e:
    print(f"   WARN — search failed: {e}")
print()

from portfolio.avanza.tick_rules import get_tick_rules, round_to_tick
print("7. Fetching tick rules...")
t0 = time.perf_counter()
try:
    rules = get_tick_rules("856394")
    dt = (time.perf_counter() - t0) * 1000
    print(f"   OK — {len(rules)} levels ({dt:.0f}ms)")
    test_price = 24.53
    rounded = round_to_tick(test_price, "856394", direction="down")
    print(f"   round_to_tick({test_price}, down) = {rounded}")
except Exception as e:
    print(f"   WARN — tick rules failed: {e}")
print()

from portfolio.avanza.trading import get_orders, get_deals
print("8. Fetching open orders...")
t0 = time.perf_counter()
try:
    orders = get_orders()
    dt = (time.perf_counter() - t0) * 1000
    print(f"   OK — {len(orders)} open orders ({dt:.0f}ms)")
except Exception as e:
    print(f"   WARN — orders failed: {e}")
print()

print("9. Fetching recent deals...")
t0 = time.perf_counter()
try:
    deals = get_deals()
    dt = (time.perf_counter() - t0) * 1000
    print(f"   OK — {len(deals)} recent deals ({dt:.0f}ms)")
    for d in deals[:3]:
        print(f"   {d.side} {d.volume}x @ {d.price:.2f} ({d.time})")
except Exception as e:
    print(f"   WARN — deals failed: {e}")
print()

print("=" * 50)
print("ALL SMOKE TESTS COMPLETED")
print("=" * 50)
