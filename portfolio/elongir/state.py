"""Persistent state management for the Elongir silver dip-trading bot."""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from portfolio.file_utils import atomic_write_json, load_json, atomic_append_jsonl

logger = logging.getLogger("portfolio.elongir.state")


# ---------------------------------------------------------------------------
# Warrant pricing helpers
# ---------------------------------------------------------------------------

def warrant_price_sek(
    silver_usd: float,
    fx_rate: float,
    financing_level: float = 75.03,
) -> float:
    """Compute MINI Long silver warrant mid-price in SEK.

    warrant = (silver_usd - financing_level) * fx_rate
    If silver <= financing_level the warrant is knocked out (value 0).
    """
    if silver_usd <= financing_level:
        return 0.0
    return (silver_usd - financing_level) * fx_rate


def effective_leverage(
    silver_usd: float,
    financing_level: float = 75.03,
) -> float:
    """Effective leverage of the MINI Long warrant.

    leverage = silver_price / (silver_price - financing_level)
    Returns infinity if at or below financing level.
    """
    diff = silver_usd - financing_level
    if diff <= 0:
        return float("inf")
    return silver_usd / diff


def buy_price(warrant_mid: float, spread_pct: float = 0.008) -> float:
    """Ask price: mid * (1 + spread/2)."""
    return warrant_mid * (1.0 + spread_pct / 2.0)


def sell_price(warrant_mid: float, spread_pct: float = 0.008) -> float:
    """Bid price: mid * (1 - spread/2)."""
    return warrant_mid * (1.0 - spread_pct / 2.0)


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """An open warrant position."""
    entry_silver_usd: float      # silver spot price at entry
    entry_warrant_sek: float     # warrant ask price at entry (SEK)
    entry_time: str              # ISO-8601 UTC
    quantity: int                # number of warrants
    cost_sek: float              # total cost including commission
    stop_price_usd: float        # hard stop on underlying silver
    trailing_peak_usd: float     # highest silver price since entry (for trailing)
    trailing_active: bool = False  # whether trailing stop is activated


# ---------------------------------------------------------------------------
# BotState
# ---------------------------------------------------------------------------

@dataclass
class BotState:
    """Complete bot state, persisted to JSON between sessions."""
    cash_sek: float = 100_000.0
    position: Optional[Position] = None
    daily_pnl: float = 0.0
    daily_trades: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    wins: int = 0
    losses: int = 0
    max_drawdown_pct: float = 0.0
    equity_peak: float = 100_000.0
    total_hold_minutes: float = 0.0
    last_trade_date: str = ""
    halted: bool = False
    halted_reason: str = ""
    signal_state: str = "SCANNING"  # SCANNING | DIP_DETECTED | CONFIRMING_BUY | IN_POSITION | TRAILING | EXIT

    def has_position(self) -> bool:
        return self.position is not None and self.position.quantity > 0

    def equity(self, silver_usd: float = 0.0, fx_rate: float = 10.5) -> float:
        """Compute current equity = cash + position mark-to-market."""
        eq = self.cash_sek
        if self.has_position() and silver_usd > 0:
            pos = self.position
            w_mid = warrant_price_sek(silver_usd, fx_rate, financing_level=75.03)
            w_bid = sell_price(w_mid)
            eq += pos.quantity * w_bid
        return eq

    def update_drawdown(self, current_equity: float) -> None:
        """Track equity peak and maximum drawdown."""
        if current_equity > self.equity_peak:
            self.equity_peak = current_equity
        if self.equity_peak > 0:
            dd = (self.equity_peak - current_equity) / self.equity_peak * 100.0
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd

    def save(self, path: str) -> None:
        """Persist state to JSON file atomically."""
        data = asdict(self)
        atomic_write_json(path, data)

    @classmethod
    def load(cls, path: str) -> "BotState":
        """Load state from JSON file. Returns default state if missing."""
        data = load_json(path)
        if data is None:
            return cls()
        pos_data = data.pop("position", None)
        position = None
        if pos_data is not None:
            position = Position(**pos_data)
        return cls(position=position, **data)

    def reset_daily(self, date_str: str) -> None:
        """Reset daily counters if it's a new day."""
        if self.last_trade_date != date_str:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.halted = False
            self.halted_reason = ""
            self.last_trade_date = date_str


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_trade(
    trades_file: str,
    action: str,
    quantity: int,
    warrant_price_sek_val: float,
    silver_usd: float,
    fx_rate: float,
    pnl_sek: float = 0.0,
    fee_sek: float = 0.0,
    reason: str = "",
) -> None:
    """Append a trade record to the JSONL trade log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "quantity": quantity,
        "warrant_price_sek": round(warrant_price_sek_val, 4),
        "silver_usd": round(silver_usd, 4),
        "fx_rate": round(fx_rate, 4),
        "pnl_sek": round(pnl_sek, 2),
        "fee_sek": round(fee_sek, 2),
        "reason": reason,
    }
    atomic_append_jsonl(trades_file, entry)


def log_poll(
    log_file: str,
    silver_usd: float,
    fx_rate: float,
    warrant_mid: float,
    signal_state: str,
    rsi_5m: Optional[float] = None,
    rsi_15m: Optional[float] = None,
    macd_hist_5m: Optional[float] = None,
    bb_pos_5m: Optional[str] = None,
    position_qty: int = 0,
    equity_sek: float = 0.0,
    leverage: Optional[float] = None,
) -> None:
    """Append a structured poll log entry to the JSONL log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "silver_usd": round(silver_usd, 4),
        "fx_rate": round(fx_rate, 4),
        "warrant_mid": round(warrant_mid, 4),
        "state": signal_state,
        "rsi_5m": round(rsi_5m, 2) if rsi_5m is not None else None,
        "rsi_15m": round(rsi_15m, 2) if rsi_15m is not None else None,
        "macd_hist_5m": round(macd_hist_5m, 6) if macd_hist_5m is not None else None,
        "bb_pos_5m": bb_pos_5m,
        "pos_qty": position_qty,
        "equity_sek": round(equity_sek, 2),
        "leverage": round(leverage, 2) if leverage is not None else None,
    }
    atomic_append_jsonl(log_file, entry)
