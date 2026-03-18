"""Elongir bot -- main decision loop for simulated silver warrant dip-trading.

Each call to step() represents one poll cycle. The bot:
1. Checks session window and daily limits
2. Computes indicators from market snapshot
3. Runs dip detector (no position) or reversal detector (in position)
4. Executes simulated buy/sell when triggered
5. Logs everything
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from portfolio.elongir.config import ElongirConfig
from portfolio.elongir.data_provider import MarketSnapshot
from portfolio.elongir.indicators import IndicatorSet, compute_all
from portfolio.elongir.signal import DipDetector, ReversalDetector
from portfolio.elongir.risk import (
    compute_position_size,
    compute_stop,
    compute_tp,
    check_daily_limits,
    check_session,
    get_stockholm_time,
)
from portfolio.elongir.state import (
    BotState,
    Position,
    warrant_price_sek,
    effective_leverage,
    buy_price,
    sell_price,
    log_trade,
    log_poll,
)

logger = logging.getLogger("portfolio.elongir.bot")


class ElongirBot:
    """Simulated silver warrant dip-trading bot.

    Stateful: maintains signal detectors, risk state, and position state
    across poll cycles.
    """

    def __init__(self, config: ElongirConfig, state: Optional[BotState] = None):
        self.cfg = config
        self.state = state or BotState.load(config.state_file)
        self.dip_detector = DipDetector(config)
        self.reversal_detector = ReversalDetector(config)
        self._current_date: Optional[str] = None

    def step(self, snapshot: MarketSnapshot) -> Optional[dict]:
        """Execute one poll cycle.

        Args:
            snapshot: Pre-collected market data.

        Returns:
            Action dict if a trade happened, None otherwise.
            Action: {"type": "BUY"|"SELL", "silver_usd": ..., "reason": ..., ...}
        """
        if not snapshot.is_complete():
            logger.warning("Incomplete snapshot -- skipping")
            return None

        # --- Daily reset ---
        _, _, date_str = get_stockholm_time()
        if self._current_date != date_str:
            self._current_date = date_str
            self.state.reset_daily(date_str)
            self.dip_detector.reset()
            logger.info("New trading day: %s -- counters reset", date_str)

        # --- Session check ---
        if not check_session(self.cfg):
            logger.debug("Outside session -- skipping")
            return None

        # --- Daily limits ---
        if self.state.halted:
            logger.debug("Trading halted: %s", self.state.halted_reason)
            return None

        ok, reason = check_daily_limits(
            self.state.daily_trades,
            self.state.daily_pnl,
            self.cfg.equity_sek,
            self.cfg,
        )
        if not ok:
            self.state.halted = True
            self.state.halted_reason = reason
            logger.warning("Trading halted: %s", reason)
            return None

        # --- Compute indicators ---
        indicators = compute_all(snapshot)

        # --- Warrant pricing ---
        w_mid = warrant_price_sek(
            snapshot.silver_usd, snapshot.fx_rate, self.cfg.financing_level
        )
        lev = effective_leverage(snapshot.silver_usd, self.cfg.financing_level)

        # --- Log poll ---
        current_equity = self.state.equity(snapshot.silver_usd, snapshot.fx_rate)
        self.state.update_drawdown(current_equity)

        log_poll(
            self.cfg.log_file,
            silver_usd=snapshot.silver_usd,
            fx_rate=snapshot.fx_rate,
            warrant_mid=w_mid,
            signal_state=self.state.signal_state,
            rsi_5m=indicators.tf_5m.rsi,
            rsi_15m=indicators.tf_15m.rsi,
            macd_hist_5m=indicators.tf_5m.macd_histogram,
            bb_pos_5m=indicators.tf_5m.bb_position,
            position_qty=self.state.position.quantity if self.state.has_position() else 0,
            equity_sek=current_equity,
            leverage=lev if lev != float("inf") else None,
        )

        # --- Decision logic ---
        if self.state.has_position():
            return self._check_position(snapshot, indicators, w_mid)
        else:
            return self._check_entry(snapshot, indicators, w_mid)

    def _check_entry(
        self,
        snapshot: MarketSnapshot,
        indicators: IndicatorSet,
        warrant_mid: float,
    ) -> Optional[dict]:
        """Run dip detector and execute buy if triggered."""
        self.state.signal_state = self.dip_detector.state

        signal = self.dip_detector.update(indicators, snapshot.silver_usd)
        self.state.signal_state = self.dip_detector.state

        if signal != "BUY":
            return None

        return self._execute_buy(snapshot, indicators, warrant_mid)

    def _check_position(
        self,
        snapshot: MarketSnapshot,
        indicators: IndicatorSet,
        warrant_mid: float,
    ) -> Optional[dict]:
        """Run reversal detector and execute sell if triggered."""
        pos = self.state.position

        # Update trailing peak
        if snapshot.silver_usd > pos.trailing_peak_usd:
            pos.trailing_peak_usd = snapshot.silver_usd

        # Check if trailing should activate
        if not pos.trailing_active:
            if self.reversal_detector.should_activate_trailing(
                snapshot.silver_usd, pos.entry_silver_usd
            ):
                pos.trailing_active = True
                self.state.signal_state = "TRAILING"
                logger.info(
                    "Trailing stop activated: silver=%.2f, entry=%.2f, peak=%.2f",
                    snapshot.silver_usd, pos.entry_silver_usd, pos.trailing_peak_usd,
                )
        else:
            self.state.signal_state = "TRAILING"

        exit_reason = self.reversal_detector.update(
            indicators,
            snapshot.silver_usd,
            pos.entry_silver_usd,
            pos.entry_time,
            pos.trailing_peak_usd,
            pos.trailing_active,
        )

        if exit_reason is not None:
            return self._execute_sell(snapshot, exit_reason, warrant_mid)

        self.state.signal_state = "IN_POSITION"
        return None

    def _execute_buy(
        self,
        snapshot: MarketSnapshot,
        indicators: IndicatorSet,
        warrant_mid: float,
    ) -> Optional[dict]:
        """Execute a simulated buy."""
        w_ask = buy_price(warrant_mid, self.cfg.spread_pct)
        if w_ask <= 0:
            logger.warning("Warrant ask price is zero -- cannot buy")
            return None

        sizing = compute_position_size(self.state.cash_sek, w_ask, self.cfg)
        if sizing.quantity <= 0:
            logger.info("Position size = 0 after sizing")
            return None

        stop = compute_stop(snapshot.silver_usd, self.cfg)
        tp = compute_tp(snapshot.silver_usd, self.cfg)

        # Update state
        self.state.cash_sek -= sizing.total_cost_sek
        self.state.total_fees += sizing.fee_sek
        self.state.daily_trades += 1
        self.state.total_trades += 1
        self.state.position = Position(
            entry_silver_usd=snapshot.silver_usd,
            entry_warrant_sek=w_ask,
            entry_time=datetime.now(timezone.utc).isoformat(),
            quantity=sizing.quantity,
            cost_sek=sizing.total_cost_sek,
            stop_price_usd=stop,
            trailing_peak_usd=snapshot.silver_usd,
            trailing_active=False,
        )
        self.state.signal_state = "IN_POSITION"
        self.state.save(self.cfg.state_file)

        # Log trade
        log_trade(
            self.cfg.trades_file,
            action="BUY",
            quantity=sizing.quantity,
            warrant_price_sek_val=w_ask,
            silver_usd=snapshot.silver_usd,
            fx_rate=snapshot.fx_rate,
            fee_sek=sizing.fee_sek,
            reason=f"DIP_BUY: RSI_5m={indicators.tf_5m.rsi:.1f}"
                   f", BB={indicators.tf_5m.bb_position}"
                   f", lev={effective_leverage(snapshot.silver_usd, self.cfg.financing_level):.1f}x"
                   if indicators.tf_5m.rsi is not None else "DIP_BUY",
        )

        action = {
            "type": "BUY",
            "silver_usd": snapshot.silver_usd,
            "warrant_ask_sek": w_ask,
            "quantity": sizing.quantity,
            "cost_sek": sizing.total_cost_sek,
            "fee_sek": sizing.fee_sek,
            "stop_usd": stop,
            "tp_usd": tp,
            "leverage": effective_leverage(snapshot.silver_usd, self.cfg.financing_level),
            "rsi_5m": indicators.tf_5m.rsi,
            "reason": "Dip detected + MACD confirmed + RSI recovery",
        }

        logger.info(
            "BUY %d warrants @ %.2f SEK (silver=$%.2f, lev=%.1fx)",
            sizing.quantity, w_ask, snapshot.silver_usd, action["leverage"],
        )
        return action

    def _execute_sell(
        self,
        snapshot: MarketSnapshot,
        reason: str,
        warrant_mid: float,
    ) -> Optional[dict]:
        """Execute a simulated sell (full exit)."""
        pos = self.state.position
        w_bid = sell_price(warrant_mid, self.cfg.spread_pct)

        if w_bid <= 0:
            # Warrant knocked out
            w_bid = 0.0

        proceeds = pos.quantity * w_bid
        fee = proceeds * self.cfg.commission_pct
        net_proceeds = proceeds - fee

        # P&L calculation
        pnl = net_proceeds - pos.cost_sek

        # Hold duration
        try:
            entry_dt = datetime.fromisoformat(pos.entry_time)
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            hold_minutes = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60.0
        except (ValueError, TypeError):
            hold_minutes = 0.0

        # Update state
        self.state.cash_sek += net_proceeds
        self.state.total_fees += fee
        self.state.daily_pnl += pnl
        self.state.total_pnl += pnl
        self.state.total_hold_minutes += hold_minutes
        if pnl >= 0:
            self.state.wins += 1
        else:
            self.state.losses += 1
        self.state.position = None
        self.state.signal_state = "SCANNING"
        self.dip_detector.reset()
        self.state.save(self.cfg.state_file)

        # Log trade
        silver_gain_pct = (snapshot.silver_usd - pos.entry_silver_usd) / pos.entry_silver_usd * 100.0
        log_trade(
            self.cfg.trades_file,
            action="SELL",
            quantity=pos.quantity,
            warrant_price_sek_val=w_bid,
            silver_usd=snapshot.silver_usd,
            fx_rate=snapshot.fx_rate,
            pnl_sek=pnl,
            fee_sek=fee,
            reason=f"{reason}: silver {silver_gain_pct:+.2f}%, held {hold_minutes:.0f}min, P&L {pnl:+,.0f} SEK",
        )

        action = {
            "type": "SELL",
            "silver_usd": snapshot.silver_usd,
            "warrant_bid_sek": w_bid,
            "quantity": pos.quantity,
            "proceeds_sek": net_proceeds,
            "fee_sek": fee,
            "pnl_sek": pnl,
            "silver_gain_pct": silver_gain_pct,
            "hold_minutes": hold_minutes,
            "reason": reason,
        }

        logger.info(
            "SELL %d warrants @ %.2f SEK (P&L: %+,.0f SEK, %s)",
            pos.quantity, w_bid, pnl, reason,
        )
        return action
