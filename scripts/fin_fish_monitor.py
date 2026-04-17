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

from portfolio.avanza_session import (
    api_get,
    api_post,
    get_instrument_price,
    get_positions,
    verify_session,
    _get_csrf,
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
# Avanza helpers (using avanza_session API functions)
# ---------------------------------------------------------------------------

def get_position_volume(ob_id: str, account_id: str) -> int:
    """Check current position volume for an instrument in an account."""
    try:
        result = api_get("/_api/position-data/positions")
        for entry in result.get("withOrderbook", []):
            inst = entry.get("instrument", {})
            ob = inst.get("orderbook", {})
            acc = entry.get("account", {})
            if str(ob.get("id", "")) == ob_id and str(acc.get("id", "")) == account_id:
                vol = entry.get("volume", {})
                return vol.get("value", 0) if isinstance(vol, dict) else (vol or 0)
    except Exception as e:
        log.warning("Position check failed: %s", e)
    return 0


def get_open_orders(ob_id: str) -> list[dict]:
    """Get open orders for a specific instrument."""
    try:
        result = api_get("/_api/trading/rest/orders")
        orders = result.get("orders", []) if isinstance(result, dict) else result
        return [
            o for o in orders
            if isinstance(o, dict)
            and str(o.get("orderbook", {}).get("id", "") or o.get("orderbookId", "")) == ob_id
        ]
    except Exception as e:
        log.warning("Order check failed: %s", e)
    return []


def get_stop_losses(ob_id: str) -> list[dict]:
    """Get active stop losses for a specific instrument."""
    try:
        result = api_get("/_api/trading/stoploss")
        if isinstance(result, list):
            return [
                s for s in result
                if isinstance(s, dict)
                and str(s.get("orderBookId", "")) == ob_id
            ]
    except Exception as e:
        log.warning("Stop loss check failed: %s", e)
    return []


def delete_stop_loss(account_id: str, stop_id: str) -> bool:
    """Delete a stop loss order."""
    try:
        from portfolio.avanza_session import api_delete
        result = api_delete(f"/_api/trading/stoploss/{stop_id}")
        ok = result.get("ok", False)
        if not ok:
            log.warning("Delete stop loss %s failed: %s", stop_id, result)
        return ok
    except Exception as e:
        log.warning("Delete stop loss %s failed: %s", stop_id, e)
        return False


def place_stop_loss(account_id: str, ob_id: str, trigger_price: float,
                    sell_price: float, volume: int, valid_days: int = 1) -> tuple[bool, str]:
    """Place a stop loss order via api_post."""
    valid_until = (
        datetime.datetime.now() + datetime.timedelta(days=valid_days)
    ).strftime("%Y-%m-%d")
    try:
        result = api_post("/_api/trading/stoploss/new", {
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
        })
        success = isinstance(result, dict) and result.get("status") == "SUCCESS"
        stop_id = result.get("stoplossOrderId", "") if isinstance(result, dict) else ""
        return success, stop_id
    except Exception as e:
        log.error("Place stop loss failed: %s", e)
        return False, ""


def fetch_price(ob_id: str) -> dict | None:
    """Fetch price info for a certificate instrument."""
    try:
        data = get_instrument_price(ob_id)
        if not data:
            return None
        listing = data.get("listing", {})
        underlying_info = data.get("underlying", {})
        key_indicators = data.get("keyIndicators", {})
        return {
            "bid": listing.get("bidPrice"),
            "ask": listing.get("askPrice"),
            "last": listing.get("lastPrice"),
            "underlying": underlying_info.get("lastPrice") if isinstance(underlying_info, dict) else None,
            "leverage": key_indicators.get("leverage", {}).get("value") if isinstance(key_indicators, dict) else None,
        }
    except Exception as e:
        log.warning("Price fetch failed: %s", e)
        return None


def force_sell(trade: FishTrade) -> bool:
    """Market sell remaining position."""
    price_info = fetch_price(trade.ob_id)
    if not price_info or not price_info.get("bid"):
        log.error("Cannot get bid price for force sell")
        return False

    bid = price_info["bid"]
    sell_price = round(bid * 0.995, 2)  # slightly below bid for guaranteed fill
    vol = trade.current_volume

    log.info("FORCE SELL %d units at %.2f (bid=%.2f)", vol, sell_price, bid)
    try:
        from portfolio.avanza_session import place_sell_order
        result = place_sell_order(trade.ob_id, sell_price, vol, trade.account_id)
        ok = isinstance(result, dict) and result.get("orderRequestStatus") == "SUCCESS"
        if ok:
            order_id = result.get("orderId", "")
            log.info("Force sell placed: order_id=%s", order_id)
            trade.closed = True
        else:
            log.error("Force sell failed: %s", result)
        return ok
    except Exception as e:
        log.error("Force sell exception: %s", e)
        return False


# ---------------------------------------------------------------------------
# Monitor loop
# ---------------------------------------------------------------------------

def monitor_trade(
    trade: FishTrade,
    poll_interval: int = 60,
    fish_interval: int = 1800,
    max_duration_hours: float = 4.0,
):
    """Main monitoring loop for an active fishing trade.

    Args:
        trade: Active trade to monitor
        poll_interval: Seconds between position checks (default 60)
        fish_interval: Seconds between full fin_fish re-runs (default 1800 = 30min)
        max_duration_hours: Auto-exit after this many hours (default 4h, prevents zombie processes)
    """
    import zoneinfo
    try:
        cet = zoneinfo.ZoneInfo("Europe/Stockholm")
    except Exception:
        import dateutil.tz
        cet = dateutil.tz.gettz("Europe/Stockholm")

    last_fish_run = 0
    check_count = 0
    monitor_start = time.time()

    # Look up instrument name from config
    from data.fin_fish_config import WARRANT_CATALOG
    inst_name = trade.ob_id
    for _key, info in WARRANT_CATALOG.items():
        if str(info.get("ob_id")) == str(trade.ob_id):
            inst_name = info.get("name", trade.ob_id)
            break

    log.info(
        "Monitoring %s | %d units @ %.2f | SL=%.2f | TP orders=%d",
        inst_name, trade.current_volume, trade.entry_price, trade.stop_trigger, len(trade.tp_orders),
    )

    try:
        while not trade.closed:
            check_count += 1
            now = datetime.datetime.now(cet)
            now_ts = time.time()

            # --- Session end check ---
            if now.hour >= 21:
                log.warning("SESSION END (21:00 CET) — force selling")
                force_sell(trade)
                break

            # --- Max duration check (prevents zombie processes) ---
            elapsed_hours = (time.time() - monitor_start) / 3600
            if elapsed_hours >= max_duration_hours:
                log.warning(
                    "MAX DURATION (%.1fh) reached — force selling to prevent zombie process",
                    max_duration_hours,
                )
                force_sell(trade)
                break

            # --- Get current state ---
            price_info = fetch_price(trade.ob_id)
            if not price_info:
                log.warning("Price fetch failed, retrying in %ds", poll_interval)
                time.sleep(poll_interval)
                continue

            bid = price_info.get("bid", 0)
            ask = price_info.get("ask", 0)
            underlying = price_info.get("underlying", 0)
            pnl = trade.pnl_pct(bid)

            # --- Check position volume ---
            vol = get_position_volume(trade.ob_id, trade.account_id)

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
                stops = get_stop_losses(trade.ob_id)
                for s in stops:
                    sid = s.get("id", "")
                    if sid:
                        log.info("Deleting old stop loss %s", sid)
                        delete_stop_loss(trade.account_id, sid)

                if vol > 0:
                    # Re-place stop loss with new volume
                    sl_ok, sl_id = place_stop_loss(
                        trade.account_id, trade.ob_id,
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
                    stops = get_stop_losses(trade.ob_id)
                    for s in stops:
                        delete_stop_loss(trade.account_id, s.get("id", ""))

                    sl_ok, sl_id = place_stop_loss(
                        trade.account_id, trade.ob_id,
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
                force_sell(trade)
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
                    # 2026-04-17 adversarial review: check exit code so
                    # crashes don't go invisible — the 3-week Layer 2
                    # auth outage used exactly this pattern (exit 0 +
                    # empty stdout + "Not logged in" on stderr).
                    if result.returncode != 0:
                        log.warning(
                            "Fin fish exited %d: stderr=%r stdout=%r",
                            result.returncode,
                            (result.stderr or "").strip()[:200],
                            (result.stdout or "").strip()[:200],
                        )
                    elif result.stdout.strip():
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
        log.info("Monitor exiting. Trade closed=%s", trade.closed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fin Fish Trade Monitor")
    parser.add_argument("--ob-id", required=True, help="Orderbook ID (e.g. 2286417)")
    parser.add_argument("--account", default="1625505", help="Account ID (default: ISK)")
    parser.add_argument("--direction", default="SHORT", choices=["LONG", "SHORT"], help="Trade direction")
    parser.add_argument("--entry-price", type=float, required=True, help="Entry price")
    parser.add_argument("--volume", type=int, required=True, help="Position volume")
    parser.add_argument("--stop-trigger", type=float, required=True, help="Stop loss trigger price")
    parser.add_argument("--tp-orders", type=str, default="[]",
                        help='TP orders as JSON array, e.g. \'[{"price":3.87,"volume":432}]\'')
    parser.add_argument("--poll", type=int, default=60, help="Poll interval seconds")
    parser.add_argument("--fish-interval", type=int, default=1800, help="Fin fish re-run interval")
    parser.add_argument("--max-duration", type=float, default=4.0,
                        help="Max monitor duration in hours (default: 4h, prevents zombie processes)")
    args = parser.parse_args()

    try:
        tp_orders = json.loads(args.tp_orders)
    except json.JSONDecodeError:
        parser.error(f"--tp-orders must be valid JSON, got: {args.tp_orders}")

    trade = FishTrade(
        ob_id=args.ob_id,
        account_id=args.account,
        direction=args.direction,
        entry_price=args.entry_price,
        volume=args.volume,
        stop_trigger=args.stop_trigger,
        tp_orders=tp_orders,
    )

    monitor_trade(
        trade,
        poll_interval=args.poll,
        fish_interval=args.fish_interval,
        max_duration_hours=args.max_duration,
    )


if __name__ == "__main__":
    main()
