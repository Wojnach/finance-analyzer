"""Warrant portfolio tracking — leverage-aware P&L for Avanza warrants.

Tracks actual warrant positions with leverage-multiplied P&L based on
the underlying instrument's price movement.
"""

import json
import logging
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.warrant_portfolio")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WARRANT_STATE_FILE = DATA_DIR / "portfolio_state_warrants.json"

_DEFAULT_STATE = {
    "holdings": {},
    "transactions": [],
}


def load_warrant_state():
    """Load warrant portfolio state from disk.

    Returns:
        dict with "holdings" and "transactions" keys.
    """
    state = load_json(WARRANT_STATE_FILE)
    if state is None:
        return _DEFAULT_STATE.copy()
    # Ensure required keys exist
    if "holdings" not in state:
        state["holdings"] = {}
    if "transactions" not in state:
        state["transactions"] = []
    return state


def save_warrant_state(state):
    """Atomically write warrant portfolio state.

    Args:
        state: dict with "holdings" and "transactions".
    """
    atomic_write_json(WARRANT_STATE_FILE, state)
    logger.info("Warrant state saved (%d holdings)", len(state.get("holdings", {})))


def warrant_pnl(holding, current_underlying_usd, fx_rate):
    """Compute P&L for a single warrant position.

    Uses the underlying price change multiplied by leverage factor.

    Args:
        holding: dict with keys:
            - units: number of warrant units held
            - entry_price_sek: price per unit at entry (SEK)
            - underlying: underlying ticker (e.g., "XAG-USD")
            - leverage: leverage factor (e.g., 5 for 5x)
            - underlying_entry_price_usd: underlying price at entry (USD)
            - name: human-readable name (optional)
        current_underlying_usd: current price of the underlying in USD.
        fx_rate: current USD/SEK exchange rate.

    Returns:
        dict: {
            "pnl_pct": float (percentage P&L),
            "pnl_sek": float (absolute P&L in SEK),
            "current_implied_sek": float (current implied value per unit),
            "total_value_sek": float (total current value),
            "entry_value_sek": float (total entry value),
            "underlying_change_pct": float (underlying price change %),
            "source": "implied"
        }
        Returns None if required data is missing.
    """
    if not holding or not current_underlying_usd or not fx_rate:
        return None

    units = holding.get("units", 0)
    entry_price_sek = holding.get("entry_price_sek", 0)
    leverage = holding.get("leverage", 1)
    underlying_entry = holding.get("underlying_entry_price_usd", 0)

    if not units or not entry_price_sek or not underlying_entry:
        return None

    # Underlying change
    underlying_change = (current_underlying_usd - underlying_entry) / underlying_entry
    underlying_change_pct = round(underlying_change * 100, 2)

    # Implied warrant P&L = underlying change * leverage
    implied_pnl_pct = underlying_change * leverage
    implied_pnl_pct_rounded = round(implied_pnl_pct * 100, 2)

    # Current implied value
    current_implied_sek = entry_price_sek * (1 + implied_pnl_pct)
    total_value_sek = current_implied_sek * units
    entry_value_sek = entry_price_sek * units
    pnl_sek = round(total_value_sek - entry_value_sek, 2)

    return {
        "pnl_pct": implied_pnl_pct_rounded,
        "pnl_sek": pnl_sek,
        "current_implied_sek": round(current_implied_sek, 2),
        "total_value_sek": round(total_value_sek, 2),
        "entry_value_sek": round(entry_value_sek, 2),
        "underlying_change_pct": underlying_change_pct,
        "source": "implied",
    }


def get_warrant_summary(prices_usd, fx_rate):
    """Build a summary of all warrant positions with current P&L.

    Args:
        prices_usd: dict {ticker: price_usd} for all instruments.
        fx_rate: current USD/SEK exchange rate.

    Returns:
        dict: {
            "positions": {
                config_key: {
                    "name": "MINI L SILVER AVA 140",
                    "underlying": "XAG-USD",
                    "leverage": 5,
                    "units": 100,
                    "pnl": { ... warrant_pnl output ... },
                }
            },
            "total_value_sek": float,
            "total_pnl_sek": float,
        }
    """
    state = load_warrant_state()
    holdings = state.get("holdings", {})

    if not holdings:
        return {"positions": {}, "total_value_sek": 0, "total_pnl_sek": 0}

    positions = {}
    total_value = 0.0
    total_pnl = 0.0

    for key, holding in holdings.items():
        underlying = holding.get("underlying")
        if not underlying:
            continue

        current_price = prices_usd.get(underlying)
        if not current_price:
            continue

        pnl = warrant_pnl(holding, current_price, fx_rate)

        position = {
            "name": holding.get("name", key),
            "underlying": underlying,
            "leverage": holding.get("leverage", 1),
            "units": holding.get("units", 0),
        }

        if pnl:
            position["pnl"] = pnl
            total_value += pnl["total_value_sek"]
            total_pnl += pnl["pnl_sek"]
        else:
            position["pnl"] = None

        positions[key] = position

    return {
        "positions": positions,
        "total_value_sek": round(total_value, 2),
        "total_pnl_sek": round(total_pnl, 2),
    }


def record_warrant_transaction(config_key, action, units, price_sek, underlying_price_usd,
                                leverage, name=None, underlying=None):
    """Record a warrant buy/sell transaction.

    Args:
        config_key: Warrant config key (e.g., "MINI-SILVER").
        action: "BUY" or "SELL".
        units: Number of units.
        price_sek: Price per unit in SEK.
        underlying_price_usd: Underlying price at transaction time.
        leverage: Leverage factor.
        name: Human-readable name (optional).
        underlying: Underlying ticker (optional).
    """
    from datetime import datetime, timezone

    state = load_warrant_state()

    txn = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_key": config_key,
        "action": action,
        "units": units,
        "price_sek": price_sek,
        "underlying_price_usd": underlying_price_usd,
        "leverage": leverage,
    }
    if name:
        txn["name"] = name
    if underlying:
        txn["underlying"] = underlying

    state["transactions"].append(txn)

    holdings = state["holdings"]
    if action == "BUY":
        if config_key in holdings:
            # Average in
            existing = holdings[config_key]
            old_units = existing.get("units", 0)
            old_price = existing.get("entry_price_sek", 0)
            new_units = old_units + units
            if new_units > 0:
                avg_price = (old_units * old_price + units * price_sek) / new_units
                existing["units"] = new_units
                existing["entry_price_sek"] = round(avg_price, 2)
        else:
            holdings[config_key] = {
                "units": units,
                "entry_price_sek": price_sek,
                "underlying": underlying or "",
                "leverage": leverage,
                "underlying_entry_price_usd": underlying_price_usd,
                "name": name or config_key,
            }
    elif action == "SELL":
        if config_key in holdings:
            existing = holdings[config_key]
            remaining = existing.get("units", 0) - units
            if remaining <= 0:
                del holdings[config_key]
            else:
                existing["units"] = remaining

    save_warrant_state(state)
    logger.info("Warrant %s %s: %d units @ %.2f SEK", action, config_key, units, price_sek)
