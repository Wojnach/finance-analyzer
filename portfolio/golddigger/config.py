"""GoldDigger configuration — all tunable parameters in one place."""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass(frozen=True)
class GolddiggerConfig:
    # --- Polling ---
    poll_seconds: int = 30
    window_n: int = 120  # rolling z-score window (120 x 30s = 60 min)
    min_window: int = 10  # minimum samples before z-score is valid

    # --- Signal weights (sum = 1.0) ---
    w_gold: float = 0.50
    w_fx: float = 0.30
    w_yield: float = 0.20

    # --- Entry / exit thresholds ---
    theta_in: float = 1.0     # composite score entry threshold (1 sigma)
    theta_out: float = 0.2    # composite score exit threshold
    confirm_polls: int = 2    # consecutive polls above theta_in to enter

    # --- Risk parameters ---
    risk_fraction: float = 0.005       # 0.5% equity risked per trade
    max_notional_fraction: float = 0.10  # max 10% equity per position
    stop_loss_pct: float = 0.05        # 5% cert drop = stop (0.25% underlying)
    take_profit_pct: float = 0.08      # 8% cert gain = TP (0.40% underlying)
    daily_loss_limit: float = 0.015    # -1.5% equity halts trading for the day
    spread_max: float = 0.02           # 2% spread hard cap (reject entry)
    max_positions: int = 1

    # --- Session window (Stockholm time) ---
    session_start_hour: int = 9
    session_start_minute: int = 2
    session_end_hour: int = 17
    session_end_minute: int = 20
    # Avanza commodity warrants: 08:15-21:55 CET, but gold certs follow
    # standard equity hours. Using conservative 09:02-17:20.

    # --- Avanza instrument ---
    bull_orderbook_id: str = ""   # BULL GULD X20 AVA orderbook ID
    bear_orderbook_id: str = ""   # BEAR GULD X20 AVA (future use)
    avanza_account_id: str = ""
    cert_api_type: str = "warrant"  # Avanza API type for price fetch

    # --- Equity ---
    equity_sek: float = 100_000.0

    # --- Data sources ---
    binance_gold_symbol: str = "XAUUSDT"  # Binance FAPI
    fred_api_key: str = ""
    fred_series: str = "DGS10"  # US 10Y yield

    # --- Kill switch ---
    kill_switch_file: str = str(DATA_DIR / "golddigger_kill")

    # --- File paths ---
    state_file: str = str(DATA_DIR / "golddigger_state.json")
    log_file: str = str(DATA_DIR / "golddigger_log.jsonl")
    trades_file: str = str(DATA_DIR / "golddigger_trades.jsonl")

    # --- Telegram ---
    telegram_alerts: bool = True
    max_daily_trades: int = 10

    @classmethod
    def from_config(cls, config: dict) -> "GolddiggerConfig":
        """Build from the main config.json's 'golddigger' section."""
        gd = config.get("golddigger", {})
        avanza = config.get("avanza", {})
        return cls(
            poll_seconds=gd.get("poll_seconds", 30),
            window_n=gd.get("window_n", 120),
            w_gold=gd.get("w_gold", 0.50),
            w_fx=gd.get("w_fx", 0.30),
            w_yield=gd.get("w_yield", 0.20),
            theta_in=gd.get("theta_in", 1.0),
            theta_out=gd.get("theta_out", 0.2),
            confirm_polls=gd.get("confirm_polls", 2),
            risk_fraction=gd.get("risk_fraction", 0.005),
            max_notional_fraction=gd.get("max_notional_fraction", 0.10),
            stop_loss_pct=gd.get("stop_loss_pct", 0.05),
            take_profit_pct=gd.get("take_profit_pct", 0.08),
            daily_loss_limit=gd.get("daily_loss_limit", 0.015),
            spread_max=gd.get("spread_max", 0.02),
            max_positions=gd.get("max_positions", 1),
            bull_orderbook_id=str(gd.get("bull_orderbook_id", "")),
            bear_orderbook_id=str(gd.get("bear_orderbook_id", "")),
            avanza_account_id=str(avanza.get("account_id", gd.get("account_id", ""))),
            cert_api_type=gd.get("cert_api_type", "warrant"),
            equity_sek=gd.get("equity_sek", 100_000.0),
            binance_gold_symbol=gd.get("binance_gold_symbol", "XAUUSDT"),
            fred_api_key=gd.get("fred_api_key", config.get("fred_api_key", "")),
            fred_series=gd.get("fred_series", "DGS10"),
            kill_switch_file=gd.get("kill_switch_file", str(DATA_DIR / "golddigger_kill")),
            state_file=gd.get("state_file", str(DATA_DIR / "golddigger_state.json")),
            log_file=gd.get("log_file", str(DATA_DIR / "golddigger_log.jsonl")),
            trades_file=gd.get("trades_file", str(DATA_DIR / "golddigger_trades.jsonl")),
            telegram_alerts=gd.get("telegram_alerts", True),
            max_daily_trades=gd.get("max_daily_trades", 10),
        )
