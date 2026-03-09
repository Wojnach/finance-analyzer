"""GoldDigger — Intraday trading bot for Avanza 20x leveraged gold certificates.

Monitors XAUUSD (gold spot), USD/SEK (FX), and US10Y yield to compute a
composite z-score signal. Trades BULL GULD X20 AVA certificates on Avanza
during Swedish market hours (09:02-17:20 CET).

Usage:
    python -m portfolio.golddigger              # run live loop
    python -m portfolio.golddigger --dry-run    # paper trade (no orders)
    python -m portfolio.golddigger --backtest   # backtest mode
"""

from portfolio.golddigger.config import GolddiggerConfig
from portfolio.golddigger.signal import CompositeSignal
from portfolio.golddigger.risk import RiskManager
from portfolio.golddigger.state import BotState
from portfolio.golddigger.bot import GolddiggerBot

__all__ = [
    "GolddiggerConfig",
    "CompositeSignal",
    "RiskManager",
    "BotState",
    "GolddiggerBot",
]
