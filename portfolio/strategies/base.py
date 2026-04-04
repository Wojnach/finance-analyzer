"""Base protocol and shared data for strategy plugins."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class SharedData:
    """Thread-safe data snapshot shared from metals loop to strategies.

    Updated by metals loop main thread, read by orchestrator thread.
    Python GIL guarantees atomic dict reads.
    """
    underlying_prices: dict[str, float] = field(default_factory=dict)
    fx_rate: float = 0.0
    cert_prices: dict[str, dict] = field(default_factory=dict)
    is_market_hours: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_price(self, ticker: str) -> float:
        """Get underlying price, 0.0 if missing."""
        return self.underlying_prices.get(ticker, 0.0)

    def get_cert(self, orderbook_id: str) -> dict | None:
        """Get certificate price data, None if missing."""
        return self.cert_prices.get(orderbook_id)


class StrategyBase(ABC):
    """Protocol for strategy plugins run by the StrategyOrchestrator."""

    @abstractmethod
    def name(self) -> str:
        """Unique strategy name for logging and config."""

    @abstractmethod
    def poll_interval_seconds(self) -> float:
        """Desired tick interval in seconds."""

    @abstractmethod
    def tick(self, shared: SharedData) -> dict | None:
        """Execute one poll cycle.

        Returns action dict if a trade happened, None otherwise.
        Must not call Playwright or block for more than a few seconds.
        """

    @abstractmethod
    def is_active(self) -> bool:
        """Whether this strategy should be ticked."""

    @abstractmethod
    def status_summary(self) -> str:
        """One-line status for Telegram/logging."""
