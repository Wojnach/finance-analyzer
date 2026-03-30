"""Unified Avanza API package.

    from portfolio.avanza import get_quote, place_order, get_positions
"""

# Auth & client
from portfolio.avanza.auth import AvanzaAuth, AuthError
from portfolio.avanza.client import AvanzaClient

# Account
from portfolio.avanza.account import get_buying_power, get_positions, get_transactions

# Market data
from portfolio.avanza.market_data import get_instrument_info, get_market_data, get_news, get_ohlc, get_quote

# Search
from portfolio.avanza.search import find_certificates, find_warrants, search

# Tick rules
from portfolio.avanza.tick_rules import clear_cache as clear_tick_cache, get_tick_rules, round_to_tick

# Trading
from portfolio.avanza.trading import (
    cancel_order, delete_stop_loss, get_deals, get_orders, get_stop_losses,
    modify_order, place_order, place_stop_loss, place_trailing_stop,
)

__all__ = [
    "AvanzaAuth", "AuthError", "AvanzaClient",
    "get_positions", "get_buying_power", "get_transactions",
    "get_quote", "get_market_data", "get_ohlc", "get_instrument_info", "get_news",
    "search", "find_warrants", "find_certificates",
    "get_tick_rules", "round_to_tick", "clear_tick_cache",
    "place_order", "modify_order", "cancel_order",
    "get_orders", "get_deals",
    "place_stop_loss", "place_trailing_stop", "get_stop_losses", "delete_stop_loss",
]
