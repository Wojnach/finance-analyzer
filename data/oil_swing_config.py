"""Oil swing trader configuration — WTI warrant trading parameters.

Mirrors `data/crypto_swing_config.py` (which itself mirrors metals_swing_config)
for the oil subsystem. All tunable constants for the autonomous WTI swing
trader live here.

Design notes:
- Universe is **WTI only** in v1. Brent (BZ=F) is deferred until Avanza
  warrant universe for Brent is verified — initial scrape of
  `data/avanza_instruments_live.json` shows OLJA-tagged warrants but
  category metadata doesn't reliably distinguish WTI from Brent. Since
  most Avanza "OLJA" certificates track Brent under the hood, we may
  pivot to Brent-as-canonical in a follow-up; for now the underlying is
  identified as "OIL-USD" and `oil_precompute.py` already maintains both.
- Ships in DRY_RUN=True. Do not flip until live warrant discovery has run
  successfully via `data/oil_warrant_refresh.py` AND the user has verified
  at least one round-trip on a tracker / MINI L cert with tight spread.
- Defaults adopted from the metals/crypto pattern verbatim. Oil-specific
  tweaks: wider stops (oil intraday wicks rival metals), 3% TP underlying
  (≈ 9% on a 3x cert).
"""

# ---------------------------------------------------------------------------
# Universe — instruments managed by the oil swing trader
# ---------------------------------------------------------------------------
INSTRUMENTS = ("OIL-USD",)

# Per-instrument data sources. yfinance is the canonical WTI/Brent feed
# (Binance does not have CL=F or BZ=F). `oil_precompute.py` already routes
# through the same source.
DATA_SOURCES = {
    "OIL-USD": {
        "yfinance_symbol": "CL=F",      # WTI front-month
        "yfinance_brent": "BZ=F",       # Brent front-month (informational)
    },
}

# ---------------------------------------------------------------------------
# Warrant catalog — populated dynamically by data/oil_warrant_refresh.py
# ---------------------------------------------------------------------------
# Static fallback used only on initial bootstrap. Live discovery overwrites
# data/oil_warrant_catalog.json every TTL_HOURS. The seed entries below come
# from data/avanza_instruments_live.json (OLJA category, scraped 2026-04-30)
# — barriers/leverage may be stale and will be refilled on first probe.
WARRANT_CATALOG_FALLBACK = {
    # Mini-Long warrants (LONG, with knockout barrier)
    "MINI_L_OLJA_AVA_624": {
        "ob_id": "2370189",
        "api_type": "warrant",
        "underlying": "OIL-USD",
        "direction": "LONG",
        "leverage": None,           # Probe will fill
        "barrier": None,
        "parity": 1,
        "name": "MINI L OLJA AVA 624",
    },
    "MINI_L_OLJA_AVA_479": {
        "ob_id": "1329405",
        "api_type": "warrant",
        "underlying": "OIL-USD",
        "direction": "LONG",
        "leverage": 1.52,
        "barrier": None,
        "parity": 1,
        "name": "MINI L OLJA AVA 479",
    },
    # Mini-Short warrants (SHORT)
    "MINI_S_OLJA_AVA_699": {
        "ob_id": "2368945",
        "api_type": "warrant",
        "underlying": "OIL-USD",
        "direction": "SHORT",
        "leverage": 2.27,
        "barrier": None,
        "parity": 1,
        "name": "MINI S OLJA AVA 699",
    },
    "MINI_S_OLJA_AVA_701": {
        "ob_id": "2368906",
        "api_type": "warrant",
        "underlying": "OIL-USD",
        "direction": "SHORT",
        "leverage": 2.55,
        "barrier": None,
        "parity": 1,
        "name": "MINI S OLJA AVA 701",
    },
    # Bear certificate (SHORT, 3x leveraged)
    "BEAR_OLJAB_X3_AVA_2": {
        "ob_id": "2367789",
        "api_type": "certificate",
        "underlying": "OIL-USD",
        "direction": "SHORT",
        "leverage": 3.0,
        "barrier": None,
        "parity": 1,
        "name": "BEAR OLJAB X3 AVA 2",
    },
}

# ---------------------------------------------------------------------------
# Execution mode
# ---------------------------------------------------------------------------
DRY_RUN = True                # Default ON — flip only after live warrant
                              # discovery + manual verification.
BUY_MODE = "AUTO"             # "AUTO" or "CONFIRM" (Telegram confirmation)
SELL_MODE = "AUTO"            # SELLs always autonomous

# ---------------------------------------------------------------------------
# Sizing & limits
# ---------------------------------------------------------------------------
ACCOUNT_ID = "1625505"        # Cash-only Avanza account (per CLAUDE.md)
INITIAL_BUDGET_SEK = 10000    # Fallback budget if Avanza API doesn't return cash
POSITION_SIZE_PCT = 25        # % of available cash per oil trade
                              # (matches crypto — oil also has wider wicks)
MAX_CONCURRENT = 1            # Single oil position at a time in v1
                              # (commodities have higher single-event risk
                              # — OPEC/inventory surprise)
TARGET_LEVERAGE = 3.0         # Preferred leverage; OLJA certs typically 1-3x
MIN_ACCEPTABLE_LEVERAGE = 1.0 # Allow 1x trackers if they exist
MIN_BARRIER_DISTANCE_PCT = 12 # Oil daily wicks ~3-5% — 12% buffer
MIN_SPREAD_PCT = 1.5          # Tighter than crypto; OLJA MINIs already tight
                              # at ~0.04% per existing scrape data
MIN_TRADE_SEK = 1000          # Avanza min courtage threshold

# ---------------------------------------------------------------------------
# Entry rules — mirror crypto_swing_config defaults
# ---------------------------------------------------------------------------
MIN_BUY_VOTERS = 3
MIN_BUY_CONFIDENCE = 0.60     # Per user rule: no sub-60% trades
MIN_BUY_TF_RATIO = 0.43       # 3/7 timeframes must agree
RSI_ENTRY_LOW = 35
RSI_ENTRY_HIGH = 68
MACD_IMPROVING_CHECKS = 2
REGIME_CONFIRM_CHECKS = 2

# Momentum-entry override
MOMENTUM_ENTRY_ENABLED = True
MOMENTUM_MIN_BUY_CONFIDENCE = 0.50
MOMENTUM_MIN_BUY_VOTERS = 2
MOMENTUM_CANDIDATE_TTL_SEC = 300
MOMENTUM_STATE_FILE = "data/oil_momentum_state.json"

# Entry-gate hardening
SIGNAL_PERSISTENCE_CHECKS = 2
MACD_DECAY_PEAK_LOOKBACK = 20
MACD_DECAY_MIN_RATIO = 0.30
RSI_SLOPE_LOOKBACK_CHECKS = 5
RSI_DIP_LOOKBACK_CHECKS = 20
RSI_DIP_BELOW_LEVEL = 55
CONFIDENCE_HISTORY_MAX = 20
RSI_HISTORY_MAX = 30

# Stale-signal rejection
MAX_SIGNAL_AGE_SEC = 900      # 15 min — same tolerance as metals/crypto

# ---------------------------------------------------------------------------
# Exit rules — oil-specific (between metals and crypto in volatility)
# ---------------------------------------------------------------------------
TAKE_PROFIT_UNDERLYING_PCT = 3.0   # Oil intraday rarely > 4% — 3% TP captures
                                    # most one-direction days. ≈ 9% on 3x cert.
TRAILING_START_PCT = 1.5           # Activate trail after 1.5% gain
TRAILING_DISTANCE_PCT = 1.0        # Trail 1% behind underlying peak
HARD_STOP_UNDERLYING_PCT = 2.5     # -2.5% underlying = hard exit
                                    # (between metals -2 and crypto -3)
SIGNAL_REVERSAL_EXIT = True

# Warrant-side exit rules
WARRANT_TAKE_PROFIT_PCT = 7.0
WARRANT_TRAILING_START_PCT = 4.0
WARRANT_TRAILING_DISTANCE_PCT = 1.5

# Momentum-exit tuning
MOMENTUM_EXIT_MIN_HOLD_SECONDS = 300
MOMENTUM_EXIT_THRESHOLD_PCT = 1.0  # Between metals 0.8 and crypto 1.2

# Hold horizons — oil futures roll, but the cert tracks front-month so we
# treat it as a continuous instrument with a 48h safety net.
MAX_HOLD_HOURS = 48                # 2-day safety net
EOD_EXIT_MINUTES_BEFORE = 0        # Oil futures trade nearly 24/7 (Sun 23:00
                                    # CET to Fri 22:00 CET on CME), so we use
                                    # MAX_HOLD_HOURS as the only time-based
                                    # exit. EOD anchor not applicable.

# ---------------------------------------------------------------------------
# Cooldowns
# ---------------------------------------------------------------------------
BUY_COOLDOWN_MINUTES = 30
LOSS_ESCALATION = {0: 1, 1: 2, 2: 4, 3: 8}

# ---------------------------------------------------------------------------
# Stop-loss (hardware, on Avanza)
# ---------------------------------------------------------------------------
STOP_LOSS_UNDERLYING_PCT = 3.0     # Oil overnight gaps possible (Mid-East,
                                    # OPEC); slightly wider than metals 2.5
STOP_LOSS_VALID_DAYS = 8

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
TELEGRAM_SUMMARY_INTERVAL = 20

# ---------------------------------------------------------------------------
# Fast-tick monitor (analog of crypto_loop's fast tick)
# ---------------------------------------------------------------------------
FAST_TICK_INTERVAL_SEC = 10        # Per-cycle sub-poll cadence
FAST_TICK_DIP_ALERT_PCT = -2.5     # Threshold for sharp-dip Telegram alert
FAST_TICK_FLUSH_PCT = 1.5          # Velocity-flush threshold
FAST_TICK_FLUSH_WINDOW_SEC = 180

# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------
STATE_FILE = "data/oil_swing_state.json"
DECISIONS_LOG = "data/oil_swing_decisions.jsonl"
TRADES_LOG = "data/oil_swing_trades.jsonl"
VALUE_HISTORY_LOG = "data/oil_value_history.jsonl"
SIGNAL_LOG = "data/oil_signal_log.jsonl"
SIGNAL_OUTCOMES_LOG = "data/oil_signal_outcomes.jsonl"
RISK_FILE = "data/oil_risk.json"
DEEP_CONTEXT_FILE = "data/oil_deep_context.json"
WARRANT_CATALOG_FILE = "data/oil_warrant_catalog.json"
