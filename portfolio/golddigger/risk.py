"""Risk management for GoldDigger — position sizing, stops, daily limits."""

import logging
import math
from dataclasses import dataclass

from portfolio.golddigger.config import GolddiggerConfig

logger = logging.getLogger("portfolio.golddigger.risk")


@dataclass
class SizeResult:
    """Result of position sizing calculation."""
    quantity: int
    risk_budget_sek: float
    notional_sek: float
    stop_price: float
    take_profit_price: float
    reason: str = ""


class RiskManager:
    """Manages position sizing, stops, and daily loss limits."""

    def __init__(self, cfg: GolddiggerConfig):
        self.cfg = cfg
        self._daily_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._halted: bool = False
        self._last_reset_date: str | None = None

    def reset_daily(self, date_str: str):
        """Reset daily counters if it's a new trading day."""
        if self._last_reset_date != date_str:
            self._daily_pnl = 0.0
            self._daily_trade_count = 0
            self._halted = False
            self._last_reset_date = date_str
            logger.info("Daily risk counters reset for %s", date_str)

    @property
    def is_halted(self) -> bool:
        return self._halted

    def record_trade_pnl(self, pnl_sek: float):
        """Record P&L from a closed trade. Checks daily loss limit."""
        self._daily_pnl += pnl_sek
        self._daily_trade_count += 1
        loss_limit = -self.cfg.daily_loss_limit * self.cfg.equity_sek
        if self._daily_pnl <= loss_limit:
            self._halted = True
            logger.warning(
                "DAILY LOSS LIMIT HIT: P&L %.0f SEK <= %.0f SEK limit. Trading halted.",
                self._daily_pnl, loss_limit,
            )

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed right now.

        Returns (allowed, reason).
        """
        if self._halted:
            return False, f"Daily loss limit hit ({self._daily_pnl:.0f} SEK)"
        if self._daily_trade_count >= self.cfg.max_daily_trades:
            return False, f"Max daily trades reached ({self._daily_trade_count})"
        return True, "ok"

    def dynamic_stop_levels(self, entry_ask: float, atr_pct: float | None = None) -> tuple:
        """Compute ATR-based stop and TP if ATR available, else fall back to fixed."""
        if atr_pct is None or not getattr(self.cfg, 'use_dynamic_stops', False):
            stop = entry_ask * (1.0 - self.cfg.stop_loss_pct)
            tp = entry_ask * (1.0 + self.cfg.take_profit_pct)
            return stop, tp

        leverage = getattr(self.cfg, 'leverage', 20.0)
        multiplier = getattr(self.cfg, 'atr_stop_multiplier', 2.0)
        underlying_stop = multiplier * atr_pct / 100.0
        cert_stop_pct = underlying_stop * leverage
        cert_stop_pct = max(getattr(self.cfg, 'atr_stop_min_pct', 0.03),
                            min(cert_stop_pct, getattr(self.cfg, 'atr_stop_max_pct', 0.15)))

        stop = entry_ask * (1.0 - cert_stop_pct)
        tp = entry_ask * (1.0 + cert_stop_pct * 1.5)  # 1.5:1 R:R
        return stop, tp

    def size_position(self, entry_ask: float, equity_sek: float, atr_pct: float | None = None) -> SizeResult:
        """Compute position size respecting risk budget and notional cap.

        Args:
            entry_ask: Certificate ask price (SEK)
            equity_sek: Current equity in SEK
            atr_pct: Optional ATR percentage for dynamic stop calculation

        Returns:
            SizeResult with quantity and price levels.
        """
        if entry_ask <= 0:
            return SizeResult(0, 0, 0, 0, 0, reason="Invalid entry price")

        # Apply slippage buffer
        effective_entry = entry_ask * (1.0 + getattr(self.cfg, 'slippage_buffer', 0.005))

        # Stop and TP prices (dynamic or fixed)
        stop_price, tp_price = self.dynamic_stop_levels(effective_entry, atr_pct)
        per_unit_risk = effective_entry - stop_price

        if per_unit_risk <= 0:
            return SizeResult(0, 0, 0, stop_price, tp_price, reason="Zero per-unit risk")

        # Risk budget: rho * equity
        risk_budget = self.cfg.risk_fraction * equity_sek
        qty_risk = math.floor(risk_budget / per_unit_risk)

        # Notional cap: eta * equity / ask (use effective_entry for conservative sizing)
        max_notional = self.cfg.max_notional_fraction * equity_sek
        qty_notional = math.floor(max_notional / effective_entry)

        qty = min(qty_risk, qty_notional)
        qty = max(qty, 0)

        notional = qty * effective_entry

        reason = ""
        if qty == 0:
            reason = "Position too small after sizing"
        elif qty == qty_notional and qty < qty_risk:
            reason = "Capped by notional limit"
        elif qty == qty_risk:
            reason = "Sized by risk budget"

        return SizeResult(
            quantity=qty,
            risk_budget_sek=risk_budget,
            notional_sek=notional,
            stop_price=stop_price,
            take_profit_price=tp_price,
            reason=reason,
        )

    def check_stop_loss(self, current_bid: float, entry_price: float) -> bool:
        """Check if stop-loss is hit. Returns True if position should be closed."""
        if current_bid <= 0 or entry_price <= 0:
            return False
        stop = entry_price * (1.0 - self.cfg.stop_loss_pct)
        return current_bid <= stop

    def check_take_profit(self, current_bid: float, entry_price: float) -> bool:
        """Check if take-profit is hit. Returns True if position should be closed."""
        if current_bid <= 0 or entry_price <= 0:
            return False
        tp = entry_price * (1.0 + self.cfg.take_profit_pct)
        return current_bid >= tp

    def check_spread(self, bid: float, ask: float) -> tuple[bool, float]:
        """Check if spread is acceptable for entry.

        Returns (acceptable, spread_pct).
        """
        if bid <= 0 or ask <= 0:
            return False, 1.0
        spread_pct = (ask - bid) / bid
        return spread_pct <= self.cfg.spread_max, spread_pct

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def daily_trade_count(self) -> int:
        return self._daily_trade_count
