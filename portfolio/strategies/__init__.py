"""Strategy plugin framework for metals loop integration."""
from portfolio.strategies.base import SharedData, StrategyBase
from portfolio.strategies.orchestrator import StrategyOrchestrator, load_strategies

__all__ = [
    "SharedData",
    "StrategyBase",
    "StrategyOrchestrator",
    "load_strategies",
]
