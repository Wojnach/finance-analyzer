"""Quick Avanza metals position check — outputs JSON to stdout.

Usage: .venv/Scripts/python.exe scripts/avanza_metals_check.py

Fetches live positions from Avanza, filters for metals-related instruments
(silver, gold, warrants, certificates, MINIs), and prints a JSON summary.
Fast — no signal computation, just live position data.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METALS_KEYWORDS = {"silver", "gold", "guld", "xau", "xag", "precious", "metal"}

# Knocked-out warrants: near-zero price, huge loss. Dead but lingers in account.
# Detection: ANY of these conditions means knocked out:
KNOCKOUT_LOSS_PCT = -85       # total P&L worse than -85%
KNOCKOUT_DAY_DROP_PCT = -70   # single-day drop > 70% = barrier hit today
KNOCKOUT_PRICE_SEK = 2.00     # last price below this + significant loss


def _is_metals_related(name: str) -> bool:
    """Check if an instrument name is metals-related."""
    lower = name.lower()
    return any(kw in lower for kw in METALS_KEYWORDS)


def _detect_knockout(pos: dict) -> bool:
    """Detect if a warrant/turbo has been knocked out.

    Knocked-out instruments stay in the account with near-zero value.
    Any of these triggers knockout detection:
    1. Single-day drop > 70% (barrier hit today)
    2. Total loss > 85% with low price
    3. Price < 2 SEK with loss > 50%
    """
    profit_pct = pos.get("profit_pct", 0)
    last_price = pos.get("last_price", 0)
    day_change = pos.get("change_today_pct", 0)

    # Single-day crash — almost certainly a knockout event
    if day_change <= KNOCKOUT_DAY_DROP_PCT:
        return True
    # Massive total loss
    if profit_pct <= KNOCKOUT_LOSS_PCT:
        return True
    # Near-zero price with significant loss
    if last_price <= KNOCKOUT_PRICE_SEK and profit_pct <= -50:
        return True
    return False


def main():
    result = {
        "success": False,
        "active": [],
        "knocked_out": [],
        "all_positions_count": 0,
        "error": None,
    }

    try:
        from portfolio.avanza_session import get_positions, verify_session

        if not verify_session():
            result["error"] = "Avanza session expired or invalid. Run: python scripts/avanza_login.py"
            print(json.dumps(result, indent=2))
            return

        all_positions = get_positions()
        result["all_positions_count"] = len(all_positions)

        for pos in all_positions:
            name = pos.get("name", "")
            if not _is_metals_related(name):
                continue

            entry = {
                "name": name,
                "orderbook_id": pos.get("orderbook_id", ""),
                "type": pos.get("type", ""),
                "units": pos.get("volume", 0),
                "value_sek": pos.get("value", 0),
                "acquired_value_sek": pos.get("acquired_value", 0),
                "profit_sek": round(pos.get("profit", 0), 2),
                "profit_pct": round(pos.get("profit_percent", 0), 2),
                "last_price": pos.get("last_price", 0),
                "change_today_pct": pos.get("change_percent", 0),
                "currency": pos.get("currency", "SEK"),
                "account_id": pos.get("account_id", ""),
            }

            if _detect_knockout(entry):
                entry["status"] = "KNOCKED_OUT"
                result["knocked_out"].append(entry)
            else:
                entry["status"] = "ACTIVE"
                result["active"].append(entry)

        result["success"] = True

    except ImportError as e:
        result["error"] = f"Import error: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
