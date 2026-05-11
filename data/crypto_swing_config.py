"""Crypto swing trader configuration — BTC + ETH warrant trading parameters.

Mirrors `data/metals_swing_config.py` for the metals subsystem. All tunable
constants for the autonomous BTC/ETH swing trader live here. Edit here, not
in the trader.

Design notes:
- BTC and ETH share the same execution venue (Avanza) and same general
  pattern (24/7 underlying, leveraged certificates). One config covers both
  via per-instrument overrides.
- Ships in DRY_RUN=True. Do not flip until live warrant discovery has run
  successfully and the user has manually verified at least one round-trip
  on a tracker certificate (XBT-TRACKER / ETH-TRACKER).
- The historical metals_swing_config evolved through several incidents
  (2026-04-17 momentum-entry override, 2026-04-18 entry-gate hardening,
  2026-04-20 warrant-side TP, 2026-04-21 stale-signal rejection). We adopt
  those battle-tested defaults verbatim; only sizing/leverage changes for
  crypto's wider noise floor.
"""

# ---------------------------------------------------------------------------
# Universe — instruments managed by the crypto swing trader
# ---------------------------------------------------------------------------
INSTRUMENTS = ("BTC-USD", "ETH-USD")

# Per-instrument data sources (Binance spot, used by Layer 1)
DATA_SOURCES = {
    "BTC-USD": {"binance_symbol": "BTCUSDT", "fapi_symbol": "BTCUSDT"},
    "ETH-USD": {"binance_symbol": "ETHUSDT", "fapi_symbol": "ETHUSDT"},
}

# ---------------------------------------------------------------------------
# Warrant catalog — populated dynamically by data/crypto_warrant_refresh.py
# ---------------------------------------------------------------------------
# Static fallback used only on initial bootstrap. Live discovery overwrites
# data/crypto_warrant_catalog.json every TTL_HOURS. Mirrors the metals
# pattern where fin_fish_config + metals_warrant_refresh cooperate.
WARRANT_CATALOG_FALLBACK = {
    # Trackers (1x, no barrier, low courtage)
    "XBT_TRACKER_AVA": {
        "ob_id": None,                # Filled by warrant_refresh on first probe
        "api_type": "tracker",
        "underlying": "BTC-USD",
        "direction": "LONG",
        "leverage": 1.0,
        "barrier": None,
        "parity": 1,
        "name": "XBT TRACKER AVA",
    },
    "ETH_TRACKER_AVA": {
        "ob_id": None,
        "api_type": "tracker",
        "underlying": "ETH-USD",
        "direction": "LONG",
        "leverage": 1.0,
        "barrier": None,
        "parity": 1,
        "name": "ETH TRACKER AVA",
    },
}

# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------
DRY_RUN = True                # Default ON — flip only after live warrant
                              # discovery + manual verification.
BUY_MODE = "AUTO"             # "AUTO" or "CONFIRM" (Telegram confirmation for BUYs)
SELL_MODE = "AUTO"            # SELLs always autonomous

# ---------------------------------------------------------------------------
# Sizing & limits
# ---------------------------------------------------------------------------
ACCOUNT_ID = "1625505"        # Cash-only Avanza account (per CLAUDE.md)
INITIAL_BUDGET_SEK = 10000    # Fallback budget if Avanza API doesn't return cash
POSITION_SIZE_PCT = 25        # % of available cash per crypto trade
                              # (slightly lower than metals' 30% — crypto wicks
                              # are wider, so smaller initial slugs)
MAX_CONCURRENT = 2            # Max simultaneous crypto positions across BTC+ETH
TARGET_LEVERAGE = 3.0         # Preferred leverage; crypto warrants typically max 5x
MIN_ACCEPTABLE_LEVERAGE = 1.0 # Allow 1x trackers (no barrier risk on those)
MIN_BARRIER_DISTANCE_PCT = 15 # Crypto wicks bigger than metals — 15% buffer vs 10%
MIN_SPREAD_PCT = 2.0          # Crypto warrants tend to be slightly wider spread
MIN_TRADE_SEK = 1000          # Avanza min courtage threshold
# Below this cash level, MIN_TRADE_SEK acts as the position size instead of
# being a sizing floor — small accounts otherwise can't place any trade.
LOW_CASH_THRESHOLD_SEK = 10_000

# ---------------------------------------------------------------------------
# Entry rules — mirror metals_swing_config defaults verbatim
# ---------------------------------------------------------------------------
MIN_BUY_VOTERS = 3
MIN_BUY_CONFIDENCE = 0.60     # Per user rule: no sub-60% trades
MIN_BUY_TF_RATIO = 0.43       # 3/7 timeframes must agree
RSI_ENTRY_LOW = 35
RSI_ENTRY_HIGH = 68
MACD_IMPROVING_CHECKS = 1     # 2026-05-11: engine-layer persistence already filters single-cycle flips; this swing-layer check is set to 1 to avoid double-counting.
REGIME_CONFIRM_CHECKS = 1     # 2026-05-11: engine-layer persistence already filters single-cycle flips; this swing-layer check is set to 1 to avoid double-counting.

# Momentum-entry override (parallels metals 2026-04-17 fix)
MOMENTUM_ENTRY_ENABLED = True
MOMENTUM_MIN_BUY_CONFIDENCE = 0.50
MOMENTUM_MIN_BUY_VOTERS = 2
MOMENTUM_CANDIDATE_TTL_SEC = 300
MOMENTUM_STATE_FILE = "data/crypto_momentum_state.json"

# Entry-gate hardening (parallels metals 2026-04-18 fix)
# 2026-05-11: engine-layer persistence already filters single-cycle flips; this swing-layer check is set to 1 to avoid double-counting.
SIGNAL_PERSISTENCE_CHECKS = 1
MACD_DECAY_PEAK_LOOKBACK = 20
MACD_DECAY_MIN_RATIO = 0.30
RSI_SLOPE_LOOKBACK_CHECKS = 5
RSI_DIP_LOOKBACK_CHECKS = 20
RSI_DIP_BELOW_LEVEL = 55
CONFIDENCE_HISTORY_MAX = 20
RSI_HISTORY_MAX = 30

# Stale-signal rejection (parallels metals 2026-04-21 fix)
MAX_SIGNAL_AGE_SEC = 900      # 15 min — same tolerance as metals

# ---------------------------------------------------------------------------
# Exit rules — wider thresholds for crypto's larger swings
# ---------------------------------------------------------------------------
# DEPRECATED 2026-05-11 — replaced by TAKE_PROFIT_WARRANT_PCT / STOP_LOSS_WARRANT_PCT
TAKE_PROFIT_UNDERLYING_PCT = 4.0   # Crypto moves bigger; 4% underlying
                                   # ≈ +12% on 3x. Was 3% for metals.
TRAILING_START_PCT = 2.0           # Activate trail after 2% underlying gain
TRAILING_DISTANCE_PCT = 1.5        # Trail 1.5% behind underlying peak
# DEPRECATED 2026-05-11 — replaced by TAKE_PROFIT_WARRANT_PCT / STOP_LOSS_WARRANT_PCT
HARD_STOP_UNDERLYING_PCT = 3.0     # -3% underlying = hard exit
                                   # (metals = -2%; crypto noisier)

# 2026-05-11: TAKE_PROFIT and STOP_LOSS are now anchored to the leveraged
# warrant's own % change (not the underlying). On a 5x cert, +5% warrant
# is reachable intraday; +3% underlying (the old anchor) is ~15% warrant,
# which silver almost never produces inside one day.
TAKE_PROFIT_WARRANT_PCT = 5.0
STOP_LOSS_WARRANT_PCT = 30.0
SIGNAL_REVERSAL_EXIT = True

# Warrant-side exit rules (parallels metals 2026-04-20 fix)
WARRANT_TAKE_PROFIT_PCT = 8.0
WARRANT_TRAILING_START_PCT = 5.0
WARRANT_TRAILING_DISTANCE_PCT = 2.0

# Momentum-exit tuning (parallels metals 2026-04-17 fix)
MOMENTUM_EXIT_MIN_HOLD_SECONDS = 300
MOMENTUM_EXIT_THRESHOLD_PCT = 1.2  # Higher than metals 0.8% — crypto noisier

# Hold horizons — crypto is 24/7 so no EOD forced exit. Use a wide safety net.
MAX_HOLD_HOURS = 72                # 3-day safety net (no EOD anchor for 24/7)
EOD_EXIT_MINUTES_BEFORE = 0        # No EOD exit (24/7 underlying)

# ---------------------------------------------------------------------------
# Cooldowns
# ---------------------------------------------------------------------------
BUY_COOLDOWN_MINUTES = 30
LOSS_ESCALATION = {0: 1, 1: 2, 2: 4, 3: 8}

# ---------------------------------------------------------------------------
# Stop-loss (hardware, on Avanza)
# ---------------------------------------------------------------------------
STOP_LOSS_UNDERLYING_PCT = 3.5     # Crypto needs wider stops; was 2.5 for metals
STOP_LOSS_VALID_DAYS = 8

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_SUMMARY_INTERVAL = 20

# ---------------------------------------------------------------------------
# Fast-tick monitor (analog of metals_loop's _silver_fast_tick)
# ---------------------------------------------------------------------------
FAST_TICK_INTERVAL_SEC = 10        # Per-cycle sub-poll cadence
FAST_TICK_DIP_ALERT_PCT = -3.0     # Threshold for sharp-dip Telegram alert
FAST_TICK_FLUSH_PCT = 2.0          # Velocity-flush threshold (recovery in <3min)
FAST_TICK_FLUSH_WINDOW_SEC = 180

# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------
STATE_FILE = "data/crypto_swing_state.json"
DECISIONS_LOG = "data/crypto_swing_decisions.jsonl"
TRADES_LOG = "data/crypto_swing_trades.jsonl"
VALUE_HISTORY_LOG = "data/crypto_value_history.jsonl"
SIGNAL_LOG = "data/crypto_signal_log.jsonl"
SIGNAL_OUTCOMES_LOG = "data/crypto_signal_outcomes.jsonl"
RISK_FILE = "data/crypto_risk.json"
DEEP_CONTEXT_FILE = "data/crypto_deep_context.json"
WARRANT_CATALOG_FILE = "data/crypto_warrant_catalog.json"
