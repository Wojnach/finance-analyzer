"""GoldDigger strategy adapter — wraps GolddiggerBot for orchestrator integration.

Builds MarketSnapshot from SharedData + lightweight Binance fetch.
Trade actions are enqueued to metals_trade_queue.json for execution
by the metals loop's Playwright session.
"""
from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.golddigger.bot import GolddiggerBot
from portfolio.golddigger.config import DATA_DIR, GolddiggerConfig
from portfolio.golddigger.data_provider import (
    MarketSnapshot,
    fetch_gold_price,
    fetch_usdsek,
    fetch_us10y_context,
    read_event_risk,
)
from portfolio.strategies.base import SharedData, StrategyBase

logger = logging.getLogger("portfolio.strategies.golddigger")

_DEFAULT_TRADE_QUEUE = str(DATA_DIR / "metals_trade_queue.json")


class GoldDiggerStrategy(StrategyBase):
    """Adapts GolddiggerBot as a strategy plugin.

    Data flow:
    - Gold spot: fetched from Binance FAPI at each tick (5s freshness)
    - FX (USD/SEK): from SharedData (metals loop cache)
    - US10Y: from GoldDigger's own provider (15min cache)
    - Cert bid/ask: from SharedData cert_prices (metals loop cache, 60s stale)
    - Trade execution: enqueued to metals_trade_queue.json
    """

    def __init__(
        self,
        config: dict,
        trade_queue_file: str = _DEFAULT_TRADE_QUEUE,
    ):
        self._cfg = GolddiggerConfig.from_config(config)
        # dry_run=False so bot tracks position state (prevents duplicate BUY signals).
        # Actual execution goes through trade queue, not Playwright.
        self._bot = GolddiggerBot(self._cfg, dry_run=False)
        self._trade_queue_file = trade_queue_file

    def name(self) -> str:
        return "golddigger"

    def poll_interval_seconds(self) -> float:
        return float(self._cfg.poll_seconds)

    def is_active(self) -> bool:
        return True

    def status_summary(self) -> str:
        state = self._bot.state
        pos = "flat"
        if state.has_position():
            pos = f"pos={state.position.quantity}x"
        return (
            f"golddigger: {pos}, "
            f"equity={state.equity_sek:.0f}, "
            f"trades={state.daily_trades}"
        )

    def tick(self, shared: SharedData) -> dict | None:
        """One poll cycle: fetch gold, build snapshot, run bot."""
        self._shared = shared  # store ref for trade queue lock
        gold = fetch_gold_price(self._cfg.binance_gold_symbol)
        if gold is None or gold <= 0:
            gold = shared.get_price("XAU-USD")
        if gold <= 0:
            return None

        snapshot = self._build_snapshot(shared, gold)
        action = self._bot.step(snapshot)

        if action is not None and action.get("action") in ("BUY", "SELL", "FLATTEN"):
            self._enqueue_trade(action, shared)

        return action

    def _build_snapshot(self, shared: SharedData, gold_price: float) -> MarketSnapshot:
        """Build a GoldDigger MarketSnapshot from shared data + own fetches."""
        # Fetch own FX rate (cached in fx_rates module, ~daily refresh)
        fx = fetch_usdsek()
        if fx is None or fx <= 0:
            fx = shared.fx_rate if shared.fx_rate > 0 else 10.5

        rate_ctx = None
        if self._cfg.fred_api_key:
            rate_ctx = fetch_us10y_context(
                self._cfg.fred_api_key,
                source=self._cfg.rates_source,
                yfinance_ticker=self._cfg.rates_proxy_ticker,
                interval=self._cfg.rates_proxy_interval,
                lookback_bars=self._cfg.rates_proxy_lookback_bars,
                ttl_seconds=self._cfg.rates_proxy_ttl_seconds,
                max_bar_age_minutes=self._cfg.rates_proxy_max_bar_age_minutes,
                fred_series=self._cfg.fred_series,
            )

        event_ctx = None
        if self._cfg.use_event_risk_gate:
            event_ctx = read_event_risk(
                hours_before=self._cfg.event_risk_hours_before,
                hours_after=self._cfg.event_risk_hours_after,
                block_types=self._cfg.event_risk_block_types,
            )

        cert = shared.get_cert(self._cfg.bull_orderbook_id)

        now = datetime.now(UTC)
        snap = MarketSnapshot(
            ts_utc=now,
            gold=gold_price,
            usdsek=fx,
            us10y=rate_ctx["value"] if rate_ctx else 0.0,
            us10y_source=rate_ctx.get("source") if rate_ctx else None,
            us10y_change_pct=rate_ctx.get("change_pct") if rate_ctx else None,
            next_event_type=event_ctx.get("event_type") if event_ctx else None,
            next_event_hours=event_ctx.get("hours_to_event") if event_ctx else None,
            event_risk_active=bool(event_ctx and event_ctx.get("active")),
            event_risk_phase=event_ctx.get("phase") if event_ctx else None,
            data_quality="ok",
            gold_fetch_ts=now,
            fx_fetch_ts=now,
        )
        if cert:
            snap.cert_bid = cert.get("bid")
            snap.cert_ask = cert.get("ask")
            snap.cert_last = cert.get("last")
            if snap.cert_bid and snap.cert_ask and snap.cert_bid > 0:
                snap.cert_spread_pct = (snap.cert_ask - snap.cert_bid) / snap.cert_bid

        return snap

    def _enqueue_trade(self, action: dict, shared: SharedData | None = None) -> None:
        """Write trade action to metals_trade_queue.json for execution.

        Uses shared.trade_queue_lock to prevent race with metals loop's
        process_trade_queue() running on the main thread.
        """
        lock = shared.trade_queue_lock if shared else None
        if lock:
            lock.acquire()
        try:
            self._enqueue_trade_locked(action)
        finally:
            if lock:
                lock.release()

    def _enqueue_trade_locked(self, action: dict) -> None:
        """Enqueue while holding the trade queue lock."""
        queue = load_json(self._trade_queue_file, default=None)
        if queue is None:
            queue = {"version": 1, "orders": []}

        order = {
            "id": str(uuid.uuid4()),
            "source": "golddigger",
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "pending",
            "action": action.get("action", action.get("type", "?")),
            "ob_id": self._cfg.bull_orderbook_id,
            "warrant_name": f"BULL GULD X{int(self._cfg.leverage)}",
            "quantity": action.get("quantity", 0),
            "price": action.get("price", 0),
            "account_id": self._cfg.avanza_account_id,
            "reason": action.get("reason", ""),
            "strategy_data": {
                "composite_s": action.get("composite_s"),
                "z_gold": action.get("z_gold"),
            },
        }
        queue["orders"].append(order)
        atomic_write_json(self._trade_queue_file, queue)
        logger.info(
            "Enqueued %s order: %s (qty=%d, ob=%s)",
            order["action"], order["id"][:8],
            order["quantity"], order["ob_id"],
        )
