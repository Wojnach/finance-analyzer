"""Elongir configuration -- all tunable parameters in one place."""

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass(frozen=True)
class ElongirConfig:
    """Configuration for the Elongir silver dip-trading bot."""

    # --- Polling ---
    poll_seconds: int = 30

    # --- RSI thresholds ---
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    rsi_recovery: float = 35.0      # RSI crossing back above this = buy trigger

    # --- Bollinger Band zones ---
    bb_period: int = 20
    bb_std: float = 2.0

    # --- MACD improving checks ---
    macd_improving_checks: int = 3   # consecutive polls with improving MACD histogram

    # --- Dip detection ---
    min_dip_pct: float = 1.0        # minimum % drop from 1h high to qualify as dip

    # --- Warrant pricing ---
    financing_level: float = 75.03   # MINI Long silver financing level (SEK)
    spread_pct: float = 0.008        # 0.8% bid-ask spread
    commission_pct: float = 0.0025   # 0.25% commission per trade

    # --- Position sizing ---
    equity_sek: float = 100_000.0
    position_size_pct: float = 0.30  # 30% of cash per trade
    max_positions: int = 1           # max 1 concurrent position

    # --- Risk parameters ---
    stop_loss_pct: float = 2.0       # 2% stop on underlying silver price
    take_profit_pct: float = 2.0     # 2% take profit on underlying silver price
    trailing_start_pct: float = 1.5  # activate trailing stop at 1.5% gain
    trailing_distance_pct: float = 0.7  # trailing stop distance 0.7% from peak
    max_hold_hours: float = 5.0      # max hold 5 hours
    daily_loss_limit_pct: float = 3.0  # halt trading after 3% daily loss
    max_daily_trades: int = 6        # max 6 trades per day

    # --- Session window (CET) ---
    session_start_hour: int = 8
    session_start_minute: int = 30
    session_end_hour: int = 21
    session_end_minute: int = 30

    # --- Telegram ---
    telegram_report_interval: int = 3600  # hourly report (seconds)

    # --- File paths ---
    state_file: str = str(DATA_DIR / "elongir_state.json")
    log_file: str = str(DATA_DIR / "elongir_log.jsonl")
    trades_file: str = str(DATA_DIR / "elongir_trades.jsonl")

    # --- Singleton lock ---
    lock_file: str = str(DATA_DIR / "elongir.singleton.lock")

    @classmethod
    def from_config(cls, config: dict) -> "ElongirConfig":
        """Build from the main config.json's 'elongir' section."""
        el = config.get("elongir", {})
        return cls(
            poll_seconds=el.get("poll_seconds", 30),
            rsi_oversold=el.get("rsi_oversold", 30.0),
            rsi_overbought=el.get("rsi_overbought", 70.0),
            rsi_recovery=el.get("rsi_recovery", 35.0),
            bb_period=el.get("bb_period", 20),
            bb_std=el.get("bb_std", 2.0),
            macd_improving_checks=el.get("macd_improving_checks", 3),
            min_dip_pct=el.get("min_dip_pct", 1.0),
            financing_level=el.get("financing_level", 75.03),
            spread_pct=el.get("spread_pct", 0.008),
            commission_pct=el.get("commission_pct", 0.0025),
            equity_sek=el.get("equity_sek", 100_000.0),
            position_size_pct=el.get("position_size_pct", 0.30),
            max_positions=el.get("max_positions", 1),
            stop_loss_pct=el.get("stop_loss_pct", 2.0),
            take_profit_pct=el.get("take_profit_pct", 2.0),
            trailing_start_pct=el.get("trailing_start_pct", 1.5),
            trailing_distance_pct=el.get("trailing_distance_pct", 0.7),
            max_hold_hours=el.get("max_hold_hours", 5.0),
            daily_loss_limit_pct=el.get("daily_loss_limit_pct", 3.0),
            max_daily_trades=el.get("max_daily_trades", 6),
            session_start_hour=el.get("session_start_hour", 8),
            session_start_minute=el.get("session_start_minute", 30),
            session_end_hour=el.get("session_end_hour", 21),
            session_end_minute=el.get("session_end_minute", 30),
            telegram_report_interval=el.get("telegram_report_interval", 3600),
            state_file=el.get("state_file", str(DATA_DIR / "elongir_state.json")),
            log_file=el.get("log_file", str(DATA_DIR / "elongir_log.jsonl")),
            trades_file=el.get("trades_file", str(DATA_DIR / "elongir_trades.jsonl")),
            lock_file=el.get("lock_file", str(DATA_DIR / "elongir.singleton.lock")),
        )
