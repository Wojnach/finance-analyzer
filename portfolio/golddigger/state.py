"""Persistent state management for GoldDigger bot."""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from portfolio.file_utils import atomic_write_json, load_json, atomic_append_jsonl

logger = logging.getLogger("portfolio.golddigger.state")


@dataclass
class Position:
    """An open certificate position."""
    orderbook_id: str
    quantity: int
    avg_price: float       # average entry price (SEK)
    entry_gold: float      # gold price at entry
    entry_time: str        # ISO-8601 UTC
    stop_price: float
    take_profit_price: float
    side: str = "BUY"      # always BUY for bull cert in v1


@dataclass
class BotState:
    """Complete bot state, persisted to JSON between sessions."""
    equity_sek: float = 100_000.0
    cash_sek: float = 100_000.0
    position: Optional[Position] = None
    daily_pnl: float = 0.0
    daily_trades: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    last_trade_date: str = ""
    last_poll_time: str = ""
    halted: bool = False
    halted_reason: str = ""

    def has_position(self) -> bool:
        return self.position is not None and self.position.quantity > 0

    def save(self, path: str):
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

    def open_position(
        self,
        orderbook_id: str,
        quantity: int,
        price: float,
        gold_price: float,
        stop_price: float,
        tp_price: float,
        fee_sek: float = 0.0,
    ):
        """Record a new position opening."""
        cost = quantity * price + fee_sek
        self.cash_sek -= cost
        self.total_fees += fee_sek
        self.daily_trades += 1
        self.total_trades += 1
        self.position = Position(
            orderbook_id=orderbook_id,
            quantity=quantity,
            avg_price=price,
            entry_gold=gold_price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_price=stop_price,
            take_profit_price=tp_price,
        )

    def close_position(self, exit_price: float, fee_sek: float = 0.0) -> float:
        """Close the current position. Returns realized P&L in SEK."""
        if not self.has_position():
            return 0.0
        pos = self.position
        proceeds = pos.quantity * exit_price - fee_sek
        cost = pos.quantity * pos.avg_price
        pnl = proceeds - cost
        self.cash_sek += proceeds
        self.total_fees += fee_sek
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.position = None
        return pnl

    def reset_daily(self, date_str: str):
        """Reset daily counters if it's a new day."""
        if self.last_trade_date != date_str:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.halted = False
            self.halted_reason = ""
            self.last_trade_date = date_str


def log_trade(
    trades_file: str,
    action: str,
    quantity: int,
    price: float,
    gold_price: float,
    composite_s: float,
    pnl: float = 0.0,
    reason: str = "",
):
    """Append a trade record to the JSONL trade log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "quantity": quantity,
        "price_sek": price,
        "gold_usd": gold_price,
        "composite_s": round(composite_s, 4),
        "pnl_sek": round(pnl, 2),
        "reason": reason,
    }
    atomic_append_jsonl(trades_file, entry)


def log_poll(
    log_file: str,
    gold: float,
    usdsek: float,
    us10y: float,
    composite_s: float,
    z_gold: float,
    z_fx: float,
    z_yield: float,
    position_qty: int = 0,
    cert_bid: Optional[float] = None,
    cert_ask: Optional[float] = None,
    data_quality: str = "ok",
    gold_volume_ratio: Optional[float] = None,
    dxy: Optional[float] = None,
    dxy_change_pct: Optional[float] = None,
    us10y_source: Optional[str] = None,
    us10y_change_pct: Optional[float] = None,
    next_event_type: Optional[str] = None,
    next_event_hours: Optional[float] = None,
    event_risk_active: bool = False,
    event_risk_phase: Optional[str] = None,
):
    """Append a structured poll log entry to the JSONL log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "gold": gold,
        "usdsek": usdsek,
        "us10y": us10y,
        "S": round(composite_s, 4),
        "z_g": round(z_gold, 4),
        "z_f": round(z_fx, 4),
        "z_y": round(z_yield, 4),
        "pos_qty": position_qty,
        "cert_bid": cert_bid,
        "cert_ask": cert_ask,
        "quality": data_quality,
        "gold_volume_ratio": gold_volume_ratio,
        "dxy": dxy,
        "dxy_change_pct": dxy_change_pct,
        "us10y_source": us10y_source,
        "us10y_change_pct": us10y_change_pct,
        "next_event_type": next_event_type,
        "next_event_hours": next_event_hours,
        "event_risk_active": event_risk_active,
        "event_risk_phase": event_risk_phase,
    }
    atomic_append_jsonl(log_file, entry)
