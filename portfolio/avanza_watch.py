"""Enhanced watch module that reads real Avanza prices.

Maps portfolio tickers to Avanza orderbook IDs and fetches live prices.
Falls back to agent_summary.json when Avanza is unavailable.
"""

import json
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"

# Avanza orderbook IDs for tracked instruments.
# Users: fill in the correct orderbook IDs from Avanza.
# Find IDs by searching on avanza.se or using find_instrument().
TICKER_TO_ORDERBOOK: dict[str, str] = {
    "BTC-USD": "",  # e.g., Bitcoin tracker/ETP orderbook ID
    "ETH-USD": "",  # e.g., Ethereum tracker/ETP orderbook ID
    "MSTR": "",     # MicroStrategy orderbook ID
    "PLTR": "",     # Palantir orderbook ID
    "NVDA": "",     # Nvidia orderbook ID
}


def get_avanza_price(ticker: str) -> Optional[float]:
    """Get current price for a ticker from Avanza.

    Looks up the Avanza orderbook ID for the ticker and fetches
    the current price. Falls back to agent_summary.json if Avanza
    is unavailable or the orderbook ID is not configured.

    Args:
        ticker: Instrument ticker (e.g., 'BTC-USD', 'MSTR')

    Returns:
        Current price as float, or None if unavailable from both sources
    """
    orderbook_id = TICKER_TO_ORDERBOOK.get(ticker, "")

    # Try Avanza first if orderbook ID is configured
    if orderbook_id:
        try:
            from portfolio.avanza_client import get_price

            info = get_price(orderbook_id)
            last_price = info.get("lastPrice")
            if last_price is not None:
                return float(last_price)
        except Exception as e:
            print(f"  [avanza_watch] Avanza price fetch failed for {ticker}: {e}")

    # Fall back to agent_summary.json
    return _get_summary_price(ticker)


def _get_summary_price(ticker: str) -> Optional[float]:
    """Get price from agent_summary.json as fallback.

    Args:
        ticker: Instrument ticker

    Returns:
        Price in USD from the summary, or None if not available
    """
    try:
        if not AGENT_SUMMARY_FILE.exists():
            return None
        summary = json.loads(AGENT_SUMMARY_FILE.read_text(encoding="utf-8"))
        ticker_data = summary.get("signals", {}).get(ticker, {})
        price = ticker_data.get("price_usd")
        if price is not None:
            return float(price)
    except Exception as e:
        print(f"  [avanza_watch] Summary price fetch failed for {ticker}: {e}")
    return None


def avanza_positions_status() -> dict:
    """Get current positions and their values from Avanza.

    Returns a dict with:
        - positions: list of position dicts from Avanza
        - total_value: total portfolio value in SEK
        - source: 'avanza' or 'unavailable'

    Falls back gracefully if Avanza client is not configured or fails.
    """
    try:
        from portfolio.avanza_client import get_positions, get_portfolio_value

        positions = get_positions()
        total_value = get_portfolio_value()
        return {
            "positions": positions,
            "total_value": total_value,
            "source": "avanza",
        }
    except Exception as e:
        print(f"  [avanza_watch] Could not fetch Avanza positions: {e}")
        return {
            "positions": [],
            "total_value": 0.0,
            "source": "unavailable",
        }
