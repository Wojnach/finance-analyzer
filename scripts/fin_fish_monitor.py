"""Fin Fish Monitor — polls Avanza and manages active fishing trades.

Runs in background, checks every 60s:
1. Position status (still held? volume changed?)
2. TP order fills (adjust stop loss volume on partial fill)
3. Time-based exits (tighten stop at 3h, force sell at 5h/21:00)
4. Re-runs fin_fish analysis every 30min
5. Sends Telegram notifications on events

Usage:
    .venv/Scripts/python.exe scripts/fin_fish_monitor.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio.avanza_session import _get_playwright_context, api_get, close_playwright
from data.metals_avanza_helpers import (
    delete_order,
    delete_stop_loss as delete_stop_loss_helper,
    fetch_price,
    place_order,
    place_stop_loss,
    get_csrf,
)
from portfolio.file_utils import atomic_append_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fin_fish_monitor")

LOG_PATH = ROOT / "data" / "fin_fish_monitor.jsonl"

# ---------------------------------------------------------------------------
# Trade state
# ---------------------------------------------------------------------------

class FishTrade:
    """Tracks one active fishing trade."""

    def __init__(
        self,
        ob_id: str,
        account_id: str,
        direction: str,
        entry_price: float,
        volume: int,
        stop_loss_id: str = "",
        stop_trigger: float = 0,
        tp_orders: list[dict] | None = None,
        entry_time: datetime.datetime | None = None,
    ):
        self.ob_id = ob_id
        self.account_id = account_id
        self.direction = direction  # "LONG" or "SHORT"
        self.entry_price = entry_price
        self.initial_volume = volume
        self.current_volume = volume
        self.stop_loss_id = stop_loss_id
        self.stop_trigger = stop_trigger
        self.tp_orders = tp_orders or []
        self.entry_time = entry_time or datetime.datetime.now()
        self.stop_tightened = False
        self.closed = False

    def hours_held(self) -> float:
        return (datetime.datetime.now() - self.entry_time).total_seconds() / 3600

    def pnl_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0
        return (current_price - self.entry_price) / self.entry_price * 100


# ---------------------------------------------------------------------------
# Avanza helpers
# ---------------------------------------------------------------------------

def get_page():
    """Get a Playwright page for Avanza API calls."""
    ctx = _get_playwright_context()
    page = ctx.new_page()
    page.goto("https://www.avanza.se")
    return page


def get_position_volume(page, ob_id: str, account_id: str) -> int:
    """Check current position volume for an instrument in an account."""
    try:
        result = page.evaluate('''async () => {
            const resp = await fetch(
                "https://www.avanza.se/_api/position-data/positions",
                {credentials: "include"}
            );
            if (resp.status !== 200) return [];
            const data = await resp.json();
            return data.withOrderbook || [];
        }''')
        for entry in result:
            inst = entry.get("instrument", {})
            ob = inst.get("orderbook", {})
            acc = entry.get("account", {})
            if str(ob.get("id", "")) == ob_id and str(acc.get("id", "")) == account_id:
                vol = entry.get("volume", {})
                return vol.get("value", 0) if isinstance(vol, dict) else (vol or 0)
    except Exception as e:
        log.warning("Position check failed: %s", e)
    return 0


def get_open_orders(page, ob_id: str) -> list[dict]:
    """Get open orders for a specific instrument."""
    try:
        result = page.evaluate('''async () => {
            const resp = await fetch(
                "https://www.avanza.se/_api/trading/rest/orders",
                {credentials: "include"}
            );
            if (resp.status !== 200) return {"orders": []};
            return await resp.json();
        }''')
        orders = result.get("orders", [])
        return [
            o for o in orders
            if isinstance(o, dict)
            and str(o.get("orderbook", {}).get("id", "")) == ob_id
        ]
    except Exception as e:
        log.warning("Order check failed: %s", e)
    return []


def get_stop_losses(page, ob_id: str) -> list[dict]:
    """Get active stop losses for a specific instrument."""
    try:
        result = page.evaluate('''async () => {
            const resp = await fetch(
                "https://www.avanza.se/_api/trading/stoploss",
                {credentials: "include"}
            );
            if (resp.status !== 200) return [];
            return await resp.json();
        }''')
        if isinstance(result, list):
            return [
                s for s in result
                if isinstance(s, dict)
                and str(s.get("orderBookId", "")) == ob_id
            ]
    except Exception as e:
        log.warning("Stop loss check failed: %s", e)
    return []


def delete_stop_loss(page, account_id: str, stop_id: str) -> bool:
    """Delete a stop loss order using the canonical helper."""
    ok, _ = delete_stop_loss_helper(page, account_id, stop_id)
    if not ok:
        log.warning("Delete stop loss %s failed", stop_id)
    return ok


def force_sell(page, trade: FishTrade) -> bool:
    """Market sell remaining position."""
    price_info = fetch_price(page, trade.ob_id, "certificate")
    if not price_info or not price_info.get("bid"):
        log.error("Cannot get bid price for force sell")
        return False

    bid = price_info["bid"]
    sell_price = round(bid * 0.995, 2)  # slightly below bid for guaranteed fill
    vol = trade.current_volume

    log.info("FORCE SELL %d units at %.2f (bid=%.2f)", vol, sell_price, bid)
    ok, result = place_order(page, trade.account_id, trade.ob_id, "SELL", sell_price, vol)
    if ok:
        log.info("Force sell placed: order_id=%s", result.get("order_id"))
        trade.closed = True
    else:
        log.error("Force sell failed: %s", result)
    return ok


# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------

def monitor_trade(trade: FishTrade, poll_interval: int = 60, fish_interval: int = 1800):
    """Main monitoring loop for an active fishing trade.

    Args:
        trade: Active trade to monitor
        poll_interval: Seconds between position checks (default 60)
        fish_interval: Seconds between full fin_fish re-runs (default 1800 = 30min)
    """
    import zoneinfo
    try:
        cet = zoneinfo.ZoneInfo("Europe/Stockholm")
    except Exception:
        import dateutil.tz
        cet = dateutil.tz.gettz("Europe/Stockholm")

    page = get_page()
    last_fish_run = 0
    check_count = 0

    log.info(
        "Monitoring BEAR SILVER X5 AVA 12 | %d units @ %.2f | SL=%.2f | TP orders=%d",
        trade.current_volume, trade.entry_price, trade.stop_trigger, len(trade.tp_orders),
    )

    try:
        while not trade.closed:
            check_count += 1
            now = datetime.datetime.now(cet)
            now_ts = time.time()

            # --- Session end check ---
            if now.hour >= 21:
                log.warning("SESSION END (21:00 CET) — force selling")
                force_sell(page, trade)
                break

            # --- Get current state ---
            price_info = fetch_price(page, trade.ob_id, "certificate")
            if not price_info:
                log.warning("Price fetch failed, retrying in %ds", poll_interval)
                time.sleep(poll_interval)
                continue

            bid = price_info.get("bid", 0)
            ask = price_info.get("ask", 0)
            underlying = price_info.get("underlying", 0)
            pnl = trade.pnl_pct(bid)

            # --- Check position volume ---
            vol = get_position_volume(page, trade.ob_id, trade.account_id)

            if vol == 0:
                log.info("Position closed (volume=0). Trade complete.")
                trade.closed = True
                break

            if vol != trade.current_volume:
                old_vol = trade.current_volume
                trade.current_volume = vol
                log.info(
                    "Volume changed: %d → %d (partial fill). Adjusting stop loss.",
                    old_vol, vol,
                )
                # Adjust stop loss volume
                stops = get_stop_losses(page, trade.ob_id)
                for s in stops:
                    sid = s.get("id", "")
                    if sid:
                        log.info("Deleting old stop loss %s", sid)
                        delete_stop_loss(page, trade.account_id, sid)

                if vol > 0:
                    # Re-place stop loss with new volume
                    sl_ok, sl_id = place_stop_loss(
                        page, trade.account_id, trade.ob_id,
                        trigger_price=trade.stop_trigger,
                        sell_price=round(trade.stop_trigger - 0.02, 2),
                        volume=vol, valid_days=1,
                    )
                    if sl_ok:
                        trade.stop_loss_id = sl_id
                        log.info("New stop loss placed: %s for %d units at %.2f",
                                 sl_id, vol, trade.stop_trigger)
                    else:
                        log.error("Failed to re-place stop loss!")

            # --- Time-based stop tightening ---
            hours = trade.hours_held()
            if hours >= 3.0 and not trade.stop_tightened:
                # Tighten stop to -0.5% underlying (roughly entry - 2.5%)
                new_trigger = round(trade.entry_price * 0.975, 2)
                if new_trigger > trade.stop_trigger:
                    log.info("3h elapsed — tightening stop from %.2f to %.2f",
                             trade.stop_trigger, new_trigger)
                    stops = get_stop_losses(page, trade.ob_id)
                    for s in stops:
                        delete_stop_loss(page, trade.account_id, s.get("id", ""))

                    sl_ok, sl_id = place_stop_loss(
                        page, trade.account_id, trade.ob_id,
                        trigger_price=new_trigger,
                        sell_price=round(new_trigger - 0.02, 2),
                        volume=trade.current_volume, valid_days=1,
                    )
                    if sl_ok:
                        trade.stop_trigger = new_trigger
                        trade.stop_loss_id = sl_id
                        trade.stop_tightened = True

            # --- Force sell at 5h ---
            if hours >= 5.0:
                log.warning("5h MAX HOLD — force selling")
                force_sell(page, trade)
                break

            # --- Status log ---
            if check_count % 5 == 0:  # every 5 checks (~5min)
                log.info(
                    "Check #%d | Bid=%.2f Ask=%.2f | Silver=$%.2f | P&L=%.1f%% | "
                    "Vol=%d | Held=%.1fh | SL=%.2f",
                    check_count, bid, ask, underlying, pnl,
                    trade.current_volume, hours, trade.stop_trigger,
                )

            # --- Periodic fin_fish re-run ---
            if now_ts - last_fish_run > fish_interval:
                log.info("Running fin_fish analysis...")
                try:
                    import subprocess
                    result = subprocess.run(
                        [str(ROOT / ".venv/Scripts/python.exe"),
                         str(ROOT / "scripts/fin_fish.py"),
                         "--metals", "silver", "--telegram"],
                        capture_output=True, text=True, timeout=60,
                        cwd=str(ROOT),
                    )
                    if result.stdout.strip():
                        log.info("Fin fish: %s", result.stdout.strip()[:200])
                except Exception as e:
                    log.warning("Fin fish run failed: %s", e)
                last_fish_run = now_ts

            # --- Log to JSONL ---
            atomic_append_jsonl(LOG_PATH, {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "check": check_count,
                "bid": bid,
                "ask": ask,
                "underlying": underlying,
                "pnl_pct": round(pnl, 2),
                "volume": trade.current_volume,
                "hours_held": round(hours, 2),
                "stop_trigger": trade.stop_trigger,
            })

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        log.info("Monitor stopped by user")
    except Exception as e:
        log.error("Monitor error: %s", e, exc_info=True)
    finally:
        try:
            page.close()
        except Exception:
            pass
        log.info("Monitor exiting. Trade closed=%s", trade.closed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fin Fish Trade Monitor")
    parser.add_argument("--ob-id", default="2286417", help="Orderbook ID")
    parser.add_argument("--account", default="1625505", help="Account ID")
    parser.add_argument("--entry-price", type=float, default=3.60, help="Entry price")
    parser.add_argument("--volume", type=int, default=1080, help="Position volume")
    parser.add_argument("--stop-trigger", type=float, default=3.42, help="Stop loss trigger")
    parser.add_argument("--poll", type=int, default=60, help="Poll interval seconds")
    parser.add_argument("--fish-interval", type=int, default=1800, help="Fin fish re-run interval")
    args = parser.parse_args()

    trade = FishTrade(
        ob_id=args.ob_id,
        account_id=args.account,
        direction="SHORT",
        entry_price=args.entry_price,
        volume=args.volume,
        stop_trigger=args.stop_trigger,
        tp_orders=[{"price": 3.87, "volume": 432, "order_id": "864492386"}],
    )

    monitor_trade(trade, poll_interval=args.poll, fish_interval=args.fish_interval)


if __name__ == "__main__":
    main()
