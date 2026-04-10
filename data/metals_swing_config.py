"""Metals swing trader configuration — warrant catalog and trading parameters.

All tunable constants for the autonomous swing trader. Edit here, not in the trader.
"""

# ---------------------------------------------------------------------------
# Warrant catalog — all tradeable instruments with metadata
# ---------------------------------------------------------------------------
WARRANT_CATALOG = {
    "MINI_L_SILVER_SG": {
        "ob_id": "2043157",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 1.56,
        "barrier": 32.45,
        "parity": 10,
        "name": "MINI L SILVER SG",
    },
    "MINI_L_SILVER_AVA_301": {
        "ob_id": "2334960",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 6.3,
        "barrier": 75.03,
        "parity": 10,
        "name": "MINI L SILVER AVA 301",
    },
    "BEAR_SILVER_X5_AVA_12": {
        "ob_id": "2286417",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": None,
        "parity": 1,
        "name": "BEAR SILVER X5 AVA 12",
    },
}

# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------
DRY_RUN = False               # True = log decisions + Telegram without placing orders
BUY_MODE = "AUTO"             # "AUTO" or "CONFIRM" (Telegram confirmation for BUYs)
SELL_MODE = "AUTO"            # SELLs always autonomous

# ---------------------------------------------------------------------------
# Sizing & limits
# ---------------------------------------------------------------------------
ACCOUNT_ID = "1625505"
INITIAL_BUDGET_SEK = 10000    # fallback budget when Avanza API fails to return cash
POSITION_SIZE_PCT = 30        # % of available cash per trade
MAX_CONCURRENT = 2            # max simultaneous positions
TARGET_LEVERAGE = 5.0         # preferred leverage (user prefers 5x)
MIN_ACCEPTABLE_LEVERAGE = 3.0 # SKIP_BUY if best candidate is below this (fail-closed)
MIN_BARRIER_DISTANCE_PCT = 10 # minimum distance to barrier (was 15, but that excluded
                              # all high-lev AVA MINIs and forced fallbacks to trackers)
MIN_SPREAD_PCT = 1.5          # max acceptable bid-ask spread %
MIN_TRADE_SEK = 1000          # minimum trade size (Avanza min courtage threshold)

# ---------------------------------------------------------------------------
# Entry rules
# ---------------------------------------------------------------------------
MIN_BUY_VOTERS = 3            # minimum agreeing BUY signals
MIN_BUY_CONFIDENCE = 0.60     # minimum calibrated signal confidence (user rule: no sub-60% trades)
MIN_BUY_TF_RATIO = 0.43       # 3/7 timeframes must agree
RSI_ENTRY_LOW = 35            # RSI buy zone lower bound
RSI_ENTRY_HIGH = 68           # RSI buy zone upper bound (avoid overbought)
MACD_IMPROVING_CHECKS = 2     # MACD must be improving for N consecutive checks
REGIME_CONFIRM_CHECKS = 2     # require N consecutive BUY checks in same regime
                              # (rejects single-check flips from trending-down → ranging BUY)

# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------
TAKE_PROFIT_UNDERLYING_PCT = 2.0   # +2% underlying = exit (+~10% on 5x warrant)
TRAILING_START_PCT = 1.5           # start trailing after 1.5% underlying gain
TRAILING_DISTANCE_PCT = 1.0        # trail 1% behind underlying peak
HARD_STOP_UNDERLYING_PCT = 2.0     # -2% underlying = hard exit
SIGNAL_REVERSAL_EXIT = True        # exit on SELL consensus with >= MIN_BUY_VOTERS
# 2026-04-10: user removed the 5h time limit in favor of an EOD-only forced
# sell (~21:45 CET, just before US market close). Set to 24h so the safety
# net still exists for catastrophic edge cases (position orphaned by a crash)
# but intraday exits are rule-driven (TAKE_PROFIT, TRAILING, HARD_STOP, SIGNAL_REVERSAL,
# MOMENTUM, EXIT_OPTIMIZER). EOD_EXIT_MINUTES_BEFORE now fires at 21:45 CET
# (10 min before the hardcoded 21:55 close in metals_swing_trader._check_exits).
# TODO: the 21:55 close_cet in _check_exits is hardcoded — DST handling is
# deferred. The .claude/rules/metals-avanza.md rule says to pull from Avanza's
# todayClosingTime API, but no helper exists yet. Next session.
MAX_HOLD_HOURS = 24                # 24h safety net only — real time-based exit is EOD
EOD_EXIT_MINUTES_BEFORE = 10       # force exit 10 min before market close (was 55 → exit too early)

# ---------------------------------------------------------------------------
# Cooldowns
# ---------------------------------------------------------------------------
BUY_COOLDOWN_MINUTES = 30     # min time between BUYs
LOSS_ESCALATION = {0: 1, 1: 2, 2: 4, 3: 8}  # consecutive losses -> cooldown multiplier

# ---------------------------------------------------------------------------
# Stop-loss (hardware, on Avanza)
# ---------------------------------------------------------------------------
STOP_LOSS_UNDERLYING_PCT = 2.5     # stop at -2.5% underlying from entry
STOP_LOSS_VALID_DAYS = 8           # stop order validity (calendar days)

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_SUMMARY_INTERVAL = 20     # send summary every N checks (~30 min at 90s)

# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------
STATE_FILE = "data/metals_swing_state.json"
DECISIONS_LOG = "data/metals_swing_decisions.jsonl"
TRADES_LOG = "data/metals_swing_trades.jsonl"
