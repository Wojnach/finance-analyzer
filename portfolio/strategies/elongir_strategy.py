"""Elongir strategy adapter — wraps ElongirBot for orchestrator integration.

Builds MarketSnapshot from SharedData + own kline fetches.
Fully simulated — no trade queue, no Avanza execution.
"""
from __future__ import annotations

import logging

from portfolio.elongir.bot import ElongirBot
from portfolio.elongir.config import ElongirConfig
from portfolio.elongir.data_provider import MarketSnapshot, fetch_klines, fetch_usdsek
from portfolio.strategies.base import SharedData, StrategyBase

logger = logging.getLogger("portfolio.strategies.elongir")


class ElongirStrategy(StrategyBase):
    """Adapts ElongirBot as a strategy plugin.

    Data flow:
    - Silver spot: from SharedData (metals loop cache, updated every 60s)
    - FX (USD/SEK): from SharedData
    - Klines (1m/5m/15m): fetched from Binance FAPI at each tick
    - Execution: fully simulated (ElongirBot manages own state)
    """

    def __init__(self, config: dict):
        self._cfg = ElongirConfig.from_config(config)
        self._bot = ElongirBot(self._cfg)

    def name(self) -> str:
        return "elongir"

    def poll_interval_seconds(self) -> float:
        return float(self._cfg.poll_seconds)

    def is_active(self) -> bool:
        return True

    def status_summary(self) -> str:
        state = self._bot.state
        pos = "flat"
        if state.has_position():
            pos = f"pos={state.position.quantity}x"
        wr = f"{state.wins}/{state.losses}" if (state.wins + state.losses) > 0 else "0/0"
        return (
            f"elongir: {pos}, "
            f"state={state.signal_state}, "
            f"pnl={state.total_pnl:+,.0f}, "
            f"W/L={wr}"
        )

    def tick(self, shared: SharedData) -> dict | None:
        """One poll cycle: build snapshot from shared data + klines, run bot."""
        silver = shared.get_price("XAG-USD")
        if silver <= 0:
            return None

        klines_1m = fetch_klines("1m", 100)
        klines_5m = fetch_klines("5m", 60)
        klines_15m = fetch_klines("15m", 40)

        snapshot = self._build_snapshot(shared, klines_1m, klines_5m, klines_15m)
        return self._bot.step(snapshot)

    def _build_snapshot(
        self,
        shared: SharedData,
        klines_1m: list | None,
        klines_5m: list | None,
        klines_15m: list | None,
    ) -> MarketSnapshot:
        """Build an Elongir MarketSnapshot from shared data + klines."""
        # Fetch own FX rate (cached in fx_rates module)
        fx = fetch_usdsek()
        if fx is None or fx <= 0:
            fx = shared.fx_rate if shared.fx_rate > 0 else 10.5
        return MarketSnapshot(
            silver_usd=shared.get_price("XAG-USD"),
            fx_rate=fx,
            klines_1m=klines_1m,
            klines_5m=klines_5m,
            klines_15m=klines_15m,
            xag_signals=None,
        )
