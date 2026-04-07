"""Strategy orchestrator — daemon thread managing multiple strategy plugins."""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from portfolio.strategies.base import SharedData, StrategyBase

logger = logging.getLogger("portfolio.strategies.orchestrator")

# Halt a strategy after this many consecutive errors
MAX_CONSECUTIVE_ERRORS = 10


class StrategyOrchestrator:
    """Manages strategy plugins in a daemon thread.

    Each strategy is ticked at its own poll interval. Errors in one
    strategy do not affect others. The thread stops cleanly on stop().
    """

    def __init__(
        self,
        strategies: list[StrategyBase],
        shared_data: SharedData,
        send_telegram: Callable[[str], object] | None = None,
    ):
        self._strategies = strategies
        self._shared = shared_data
        self._send_telegram = send_telegram
        self._last_tick: dict[str, float] = {}
        self._error_counts: dict[str, int] = {}
        self._halted: set[str] = set()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the orchestrator daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="strategy-orchestrator",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Strategy orchestrator started: %s",
            ", ".join(s.name() for s in self._strategies),
        )

    def stop(self) -> None:
        """Signal the thread to stop and wait for it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Strategy orchestrator stopped")

    def summary(self) -> str:
        """One-line summary of all strategies."""
        parts = []
        for s in self._strategies:
            status = "HALTED" if s.name() in self._halted else (
                "active" if s.is_active() else "inactive"
            )
            parts.append(f"{s.name()}({status}, {s.poll_interval_seconds()}s)")
        return f"{len(self._strategies)} strategies: " + ", ".join(parts)

    def _run_loop(self) -> None:
        """Main tick loop — runs until stop() is called."""
        while self._running:
            now = time.monotonic()
            for strategy in self._strategies:
                name = strategy.name()

                if name in self._halted:
                    continue
                if not strategy.is_active():
                    continue

                last = self._last_tick.get(name, 0.0)
                if now - last < strategy.poll_interval_seconds():
                    continue

                try:
                    action = strategy.tick(self._shared)
                    self._last_tick[name] = time.monotonic()
                    self._error_counts[name] = 0

                    if action is not None:
                        self._handle_action(strategy, action)

                except Exception:
                    count = self._error_counts.get(name, 0) + 1
                    self._error_counts[name] = count
                    logger.error(
                        "Strategy %s error (%d/%d)",
                        name, count, MAX_CONSECUTIVE_ERRORS,
                        exc_info=True,
                    )
                    if count >= MAX_CONSECUTIVE_ERRORS:
                        self._halted.add(name)
                        logger.error(
                            "Strategy %s HALTED after %d consecutive errors",
                            name, count,
                        )
                        if self._send_telegram:
                            self._send_telegram(
                                f"_Strategy {name} halted: {count} consecutive errors_"
                            )

            time.sleep(0.5)

    def _handle_action(self, strategy: StrategyBase, action: dict) -> None:
        """Process a trade action from a strategy."""
        logger.info(
            "Strategy %s action: %s",
            strategy.name(),
            action.get("type", action.get("action", "?")),
        )
        if self._send_telegram:
            action_type = action.get("type", action.get("action", "?"))
            reason = action.get("reason", "")
            self._send_telegram(
                f"*STRATEGY {strategy.name().upper()}* {action_type}\n_{reason}_"
            )


def load_strategies(config: dict) -> list[StrategyBase]:
    """Load enabled strategies from config.

    Reads config["strategies"]["golddigger_enabled"] and
    config["strategies"]["elongir_enabled"] to decide which to load.
    Defaults to enabled if the bot's config section exists.
    """
    strategies_cfg = config.get("strategies", {})
    strategies: list[StrategyBase] = []

    # GoldDigger
    gd_enabled = strategies_cfg.get("golddigger_enabled", "golddigger" in config)
    if gd_enabled:
        try:
            from portfolio.strategies.golddigger_strategy import GoldDiggerStrategy
            strategies.append(GoldDiggerStrategy(config))
            logger.info("Loaded strategy: golddigger")
        except Exception as e:
            logger.error("Failed to load golddigger strategy: %s", e)

    # Elongir
    el_enabled = strategies_cfg.get("elongir_enabled", "elongir" in config)
    if el_enabled:
        try:
            from portfolio.strategies.elongir_strategy import ElongirStrategy
            strategies.append(ElongirStrategy(config))
            logger.info("Loaded strategy: elongir")
        except Exception as e:
            logger.error("Failed to load elongir strategy: %s", e)

    return strategies
