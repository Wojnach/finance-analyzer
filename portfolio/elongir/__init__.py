"""Elongir -- Simulated silver dip-trading agent.

Buys silver warrant dips and sells reversals from the top.
100K SEK starting capital. Hourly Telegram reports.

Usage:
    python -m portfolio.elongir              # run main loop
    python -m portfolio.elongir --once       # single poll cycle
    python -m portfolio.elongir --dry-run    # paper trade (default)
"""

from portfolio.elongir.config import ElongirConfig
from portfolio.elongir.state import BotState, Position
from portfolio.elongir.bot import ElongirBot

__all__ = [
    "ElongirConfig",
    "BotState",
    "Position",
    "ElongirBot",
]
