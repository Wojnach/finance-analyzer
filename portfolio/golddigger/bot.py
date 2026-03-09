"""GoldDigger bot — main decision loop for intraday gold certificate trading.

Each call to step() represents one 30-second poll cycle. The bot:
1. Checks kill switch and session window
2. Collects market data snapshot
3. Computes composite signal
4. Decides: enter, exit, or hold
5. Logs everything

Execution is separated: step() returns an action dict that the runner
can either execute (live) or record (dry-run/backtest).
"""

import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from portfolio.golddigger.config import GolddiggerConfig
from portfolio.golddigger.data_provider import MarketSnapshot, collect_snapshot
from portfolio.golddigger.signal import CompositeSignal, SignalState
from portfolio.golddigger.risk import RiskManager, SizeResult
from portfolio.golddigger.state import BotState, log_trade, log_poll
from portfolio.golddigger.augmented_signals import AugmentedSignals

logger = logging.getLogger("portfolio.golddigger.bot")


def _now_stockholm():
    """Get current Stockholm (CET/CEST) time as (hour, minute) tuple."""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Europe/Stockholm")
    except ImportError:
        # Fallback: assume CET (UTC+1) — close enough for market hours
        from datetime import timedelta
        class _CET(datetime.tzinfo):
            def utcoffset(self, dt): return timedelta(hours=1)
            def tzname(self, dt): return "CET"
            def dst(self, dt): return timedelta(0)
        tz = _CET()
    now = datetime.now(tz)
    return now.hour, now.minute, now.strftime("%Y-%m-%d")


class GolddiggerBot:
    """Intraday gold certificate trading bot.

    Stateful: maintains signal engine, risk manager, and position state
    across poll cycles within a single trading day.
    """

    def __init__(self, cfg: GolddiggerConfig, dry_run: bool = True):
        self.cfg = cfg
        self.dry_run = dry_run
        self.signal = CompositeSignal(
            window_n=cfg.window_n,
            min_window=cfg.min_window,
            w_gold=cfg.w_gold,
            w_fx=cfg.w_fx,
            w_yield=cfg.w_yield,
            theta_in=cfg.theta_in,
            theta_out=cfg.theta_out,
            confirm_polls=cfg.confirm_polls,
        )
        self.risk = RiskManager(cfg)
        self.state = BotState.load(cfg.state_file)
        self._current_date: Optional[str] = None
        self._page = None  # Playwright page, set externally

        # Augmented signal gates (volatility, momentum, structure)
        self.augmented = AugmentedSignals(
            symbol=cfg.binance_gold_symbol,
            lookback_bars=getattr(cfg, 'aug_kline_bars', 120),
            refresh_interval=getattr(cfg, 'aug_refresh_seconds', 60.0),
        ) if getattr(cfg, 'use_augmented_signals', False) else None

    def set_page(self, page):
        """Set the Playwright page for Avanza API calls."""
        self._page = page

    def _check_kill_switch(self) -> bool:
        """Check if kill switch file exists or env var is set."""
        if os.environ.get("GOLDDIGGER_KILL", "0") == "1":
            return True
        return Path(self.cfg.kill_switch_file).exists()

    def _in_session(self, hour: int, minute: int) -> bool:
        """Check if current Stockholm time is within trading session.

        Includes flatten time (session_end) so we can still exit positions
        at the boundary. New entries are blocked by _should_flatten().
        """
        start = self.cfg.session_start_hour * 60 + self.cfg.session_start_minute
        end = self.cfg.session_end_hour * 60 + self.cfg.session_end_minute
        now = hour * 60 + minute
        return start <= now <= end

    def _should_flatten(self, hour: int, minute: int) -> bool:
        """Check if we're at or past the flatten time (17:20 Stockholm)."""
        flatten = self.cfg.session_end_hour * 60 + self.cfg.session_end_minute
        now = hour * 60 + minute
        return now >= flatten

    def step(self, snapshot: Optional[MarketSnapshot] = None) -> Optional[dict]:
        """Execute one poll cycle.

        Args:
            snapshot: Pre-collected market data. If None, will collect live data.

        Returns:
            Action dict if a trade should happen, None otherwise.
            Action: {"action": "BUY"|"SELL"|"FLATTEN", "quantity": int,
                     "price": float, "reason": str, ...}
        """
        # --- Kill switch ---
        if self._check_kill_switch():
            logger.warning("Kill switch active — no trading")
            return None

        # --- Time checks ---
        hour, minute, date_str = _now_stockholm()

        # Daily reset
        if self._current_date != date_str:
            self._current_date = date_str
            self.signal.reset()
            self.risk.reset_daily(date_str)
            self.state.reset_daily(date_str)
            logger.info("New trading day: %s — signal and risk reset", date_str)

        # Outside session?
        if not self._in_session(hour, minute):
            logger.debug("Outside session (%02d:%02d) — skipping", hour, minute)
            return None

        # --- Collect data ---
        if snapshot is None:
            snapshot = collect_snapshot(
                fred_api_key=self.cfg.fred_api_key,
                page=self._page,
                orderbook_id=self.cfg.bull_orderbook_id,
                api_type=self.cfg.cert_api_type,
            )

        if not snapshot.is_complete():
            logger.warning("Incomplete data: quality=%s — holding", snapshot.data_quality)
            return None

        # --- Freshness check ---
        stale_max = getattr(self.cfg, 'stale_data_max_seconds', 90.0)
        if not snapshot.is_fresh(stale_max):
            logger.warning("Stale data (>%.0fs) — holding", stale_max)
            return None

        # --- Compute signal ---
        sig = self.signal.update(snapshot)

        # --- Log poll ---
        log_poll(
            self.cfg.log_file,
            gold=snapshot.gold,
            usdsek=snapshot.usdsek,
            us10y=snapshot.us10y,
            composite_s=sig.composite_s,
            z_gold=sig.z_gold,
            z_fx=sig.z_fx,
            z_yield=sig.z_yield,
            position_qty=self.state.position.quantity if self.state.has_position() else 0,
            cert_bid=snapshot.cert_bid,
            cert_ask=snapshot.cert_ask,
            data_quality=snapshot.data_quality,
        )

        self.state.last_poll_time = datetime.now(timezone.utc).isoformat()

        # --- Flatten at session end ---
        if self._should_flatten(hour, minute) and self.state.has_position():
            return self._make_exit_action(
                snapshot, sig, reason="SESSION_FLATTEN (>= 17:20 Stockholm)"
            )

        # --- If holding a position: check stops ---
        if self.state.has_position():
            return self._check_exit_conditions(snapshot, sig)

        # --- No position: check entry conditions ---
        return self._check_entry_conditions(snapshot, sig)

    def _check_entry_conditions(
        self, snap: MarketSnapshot, sig: SignalState
    ) -> Optional[dict]:
        """Check if we should enter a new position."""
        # Risk manager gate
        can_trade, reason = self.risk.can_trade()
        if not can_trade:
            logger.info("Entry blocked by risk: %s", reason)
            return None

        # Signal consensus gate (Phase 2)
        if getattr(self.cfg, 'use_signal_consensus', False):
            from portfolio.golddigger.data_provider import read_xau_consensus
            consensus = read_xau_consensus()
            if consensus and consensus["action"] == "SELL":
                logger.info("Entry blocked: XAU-USD consensus SELL (%dB/%dS)",
                             consensus.get("buy_count", 0), consensus.get("sell_count", 0))
                return None

        # DXY macro gate (Phase 2)
        if getattr(self.cfg, 'use_macro_context', False):
            from portfolio.golddigger.data_provider import read_macro_context
            macro = read_macro_context()
            dxy_change = macro.get("dxy_5d_change")
            if dxy_change is not None and dxy_change > 1.0:
                logger.info("Entry blocked: DXY strengthening +%.1f%% 5d", dxy_change)
                return None

        # Volume gate (Phase 4)
        if getattr(self.cfg, 'use_volume_confirm', False) and snap.gold_volume_ratio is not None:
            if snap.gold_volume_ratio < 0.5:
                logger.info("Entry blocked: low volume (ratio=%.2f)", snap.gold_volume_ratio)
                return None

        # Chronos forecast gate (Phase 4)
        if getattr(self.cfg, 'use_chronos_forecast', False):
            from portfolio.golddigger.data_provider import read_chronos_forecast
            forecast = read_chronos_forecast("XAU-USD")
            if forecast and forecast["action"] == "SELL" and forecast.get("confidence", 0) > 0.6:
                logger.info("Entry blocked: Chronos forecast SELL (conf=%.2f)", forecast["confidence"])
                return None

        # Spread check
        spread_pct = snap.cert_spread_pct
        if spread_pct is not None:
            ok, spread = self.risk.check_spread(
                snap.cert_bid or 0, snap.cert_ask or 0
            )
            if not ok:
                logger.info("Entry blocked: spread %.2f%% > max", spread * 100)
                return None

        # Augmented signal gates (volatility, momentum, structure from 1m klines)
        if self.augmented is not None:
            aug_state = self.augmented.refresh_if_needed()
            allowed, aug_reason = aug_state.entry_allowed(
                require_vol_confirm=getattr(self.cfg, 'aug_require_vol_confirm', True),
                block_on_momentum_sell=getattr(self.cfg, 'aug_block_momentum_sell', True),
                block_on_structure_sell=getattr(self.cfg, 'aug_block_structure_sell', True),
            )
            if not allowed:
                logger.info("Entry blocked: %s [%s]", aug_reason, aug_state.summary())
                return None

        # Signal check
        if not self.signal.should_enter(sig, spread_pct, self.cfg.spread_max):
            return None

        # ATR for dynamic stops
        atr_pct = None
        if getattr(self.cfg, 'use_dynamic_stops', False):
            from portfolio.golddigger.data_provider import read_xau_atr
            atr_pct = read_xau_atr()

        # Size the position
        entry_price = snap.cert_ask or snap.cert_last or 0
        if entry_price <= 0:
            logger.warning("No valid certificate price for entry")
            return None

        sizing = self.risk.size_position(entry_price, self.state.cash_sek, atr_pct=atr_pct)
        if sizing.quantity <= 0:
            logger.info("Position size = 0 after sizing: %s", sizing.reason)
            return None

        action = {
            "action": "BUY",
            "orderbook_id": self.cfg.bull_orderbook_id,
            "quantity": sizing.quantity,
            "price": entry_price,
            "stop_price": sizing.stop_price,
            "take_profit_price": sizing.take_profit_price,
            "gold_price": snap.gold,
            "composite_s": sig.composite_s,
            "z_gold": sig.z_gold,
            "notional_sek": sizing.notional_sek,
            "reason": f"ENTRY: S={sig.composite_s:.2f} >= {self.cfg.theta_in} "
                      f"(confirmed {sig.confirm_count}x), z_gold={sig.z_gold:.2f}"
                      + (f" | {self.augmented.state.summary()}" if self.augmented else ""),
        }

        if not self.dry_run:
            self.state.open_position(
                orderbook_id=self.cfg.bull_orderbook_id,
                quantity=sizing.quantity,
                price=entry_price,
                gold_price=snap.gold,
                stop_price=sizing.stop_price,
                tp_price=sizing.take_profit_price,
            )
            self.state.save(self.cfg.state_file)

        log_trade(
            self.cfg.trades_file,
            action="BUY",
            quantity=sizing.quantity,
            price=entry_price,
            gold_price=snap.gold,
            composite_s=sig.composite_s,
            reason=action["reason"],
        )

        logger.info("BUY %d @ %.2f SEK (S=%.2f)", sizing.quantity, entry_price, sig.composite_s)
        return action

    def _check_exit_conditions(
        self, snap: MarketSnapshot, sig: SignalState
    ) -> Optional[dict]:
        """Check stop-loss, take-profit, and signal exit for open position."""
        pos = self.state.position
        bid = snap.cert_bid or snap.cert_last or 0

        if bid <= 0:
            logger.warning("No valid certificate bid for exit check")
            return None

        # Stop-loss
        if self.risk.check_stop_loss(bid, pos.avg_price):
            return self._make_exit_action(
                snap, sig,
                reason=f"STOP_LOSS: bid {bid:.2f} <= stop {pos.stop_price:.2f}",
            )

        # Take-profit
        if self.risk.check_take_profit(bid, pos.avg_price):
            return self._make_exit_action(
                snap, sig,
                reason=f"TAKE_PROFIT: bid {bid:.2f} >= TP {pos.take_profit_price:.2f}",
            )

        # Signal exit (composite decayed)
        if self.signal.should_exit(sig):
            return self._make_exit_action(
                snap, sig,
                reason=f"SIGNAL_EXIT: S={sig.composite_s:.2f} <= {self.cfg.theta_out}",
            )

        return None

    def _make_exit_action(
        self, snap: MarketSnapshot, sig: SignalState, reason: str
    ) -> dict:
        """Build an exit action and update state."""
        pos = self.state.position
        bid = snap.cert_bid or snap.cert_last or 0

        pnl = 0.0
        if not self.dry_run:
            pnl = self.state.close_position(bid)
            self.risk.record_trade_pnl(pnl)
            self.state.save(self.cfg.state_file)

        log_trade(
            self.cfg.trades_file,
            action="SELL",
            quantity=pos.quantity,
            price=bid,
            gold_price=snap.gold,
            composite_s=sig.composite_s,
            pnl=pnl,
            reason=reason,
        )

        logger.info("SELL %d @ %.2f SEK (P&L: %.0f SEK) — %s",
                     pos.quantity, bid, pnl, reason)

        return {
            "action": "SELL",
            "orderbook_id": pos.orderbook_id,
            "quantity": pos.quantity,
            "price": bid,
            "gold_price": snap.gold,
            "composite_s": sig.composite_s,
            "pnl_sek": pnl,
            "reason": reason,
        }
