"""Position sizing and risk management for the Elongir bot."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime

from portfolio.elongir.config import ElongirConfig

logger = logging.getLogger("portfolio.elongir.risk")


@dataclass
class SizeResult:
    """Result of position sizing calculation."""
    quantity: int
    cost_sek: float          # warrant_ask * quantity
    fee_sek: float           # commission
    total_cost_sek: float    # cost + fee (deducted from cash)
    warrant_ask: float       # per-unit ask price


def compute_position_size(
    cash_sek: float,
    warrant_ask: float,
    config: ElongirConfig,
) -> SizeResult:
    """Compute how many warrants to buy.

    quantity = floor(cash * position_size_pct / warrant_ask)
    cost = quantity * warrant_ask
    fee = cost * commission_pct
    total = cost + fee
    """
    if warrant_ask <= 0:
        return SizeResult(0, 0.0, 0.0, 0.0, warrant_ask)

    allocation = cash_sek * config.position_size_pct
    raw_qty = allocation / warrant_ask
    quantity = math.floor(raw_qty)

    if quantity <= 0:
        return SizeResult(0, 0.0, 0.0, 0.0, warrant_ask)

    cost = quantity * warrant_ask
    fee = cost * config.commission_pct
    total = cost + fee

    # Verify we can afford it
    if total > cash_sek:
        # Reduce quantity until it fits
        quantity = math.floor((cash_sek / (1.0 + config.commission_pct)) / warrant_ask)
        if quantity <= 0:
            return SizeResult(0, 0.0, 0.0, 0.0, warrant_ask)
        cost = quantity * warrant_ask
        fee = cost * config.commission_pct
        total = cost + fee

    return SizeResult(
        quantity=quantity,
        cost_sek=round(cost, 2),
        fee_sek=round(fee, 2),
        total_cost_sek=round(total, 2),
        warrant_ask=warrant_ask,
    )


def compute_stop(
    entry_silver_usd: float,
    config: ElongirConfig,
) -> float:
    """Compute hard stop price on underlying silver.

    stop = entry * (1 - stop_loss_pct / 100)
    """
    return entry_silver_usd * (1.0 - config.stop_loss_pct / 100.0)


def compute_tp(
    entry_silver_usd: float,
    config: ElongirConfig,
) -> float:
    """Compute take-profit price on underlying silver.

    tp = entry * (1 + take_profit_pct / 100)
    """
    return entry_silver_usd * (1.0 + config.take_profit_pct / 100.0)


def check_daily_limits(
    daily_trades: int,
    daily_pnl: float,
    equity_sek: float,
    config: ElongirConfig,
) -> tuple[bool, str]:
    """Check if trading is allowed given daily limits.

    Returns (ok, reason). ok=True means trading allowed.
    """
    if daily_trades >= config.max_daily_trades:
        return False, f"Max daily trades reached ({daily_trades}/{config.max_daily_trades})"

    loss_limit = equity_sek * config.daily_loss_limit_pct / 100.0
    if daily_pnl < 0 and abs(daily_pnl) >= loss_limit:
        return False, f"Daily loss limit reached ({daily_pnl:.0f} SEK >= {loss_limit:.0f} SEK)"

    return True, "OK"


def check_session(config: ElongirConfig) -> bool:
    """Check if current CET time is within the configured trading session.

    Returns True if within session hours.
    """
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Europe/Stockholm")
    except ImportError:
        from datetime import timedelta

        class _CET(datetime.tzinfo):
            def utcoffset(self, dt):
                return timedelta(hours=1)
            def tzname(self, dt):
                return "CET"
            def dst(self, dt):
                return timedelta(0)
        tz = _CET()

    now = datetime.now(tz)
    current_minutes = now.hour * 60 + now.minute
    start_minutes = config.session_start_hour * 60 + config.session_start_minute
    end_minutes = config.session_end_hour * 60 + config.session_end_minute

    return start_minutes <= current_minutes <= end_minutes


def get_stockholm_time() -> tuple[int, int, str]:
    """Get current Stockholm (CET/CEST) time as (hour, minute, date_str)."""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("Europe/Stockholm")
    except ImportError:
        from datetime import timedelta

        class _CET(datetime.tzinfo):
            def utcoffset(self, dt):
                return timedelta(hours=1)
            def tzname(self, dt):
                return "CET"
            def dst(self, dt):
                return timedelta(0)
        tz = _CET()

    now = datetime.now(tz)
    return now.hour, now.minute, now.strftime("%Y-%m-%d")
