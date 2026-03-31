"""Fin-fish configuration — warrant catalog and fishing parameters.

All tunable constants for the dip/peak fishing bot. Edit here, not in the trader.
"""

# ---------------------------------------------------------------------------
# Warrant catalog — all tradeable instruments with metadata
# ---------------------------------------------------------------------------
WARRANT_CATALOG = {
    # -----------------------------------------------------------------------
    # Silver LONG (fish dips — buy BULL cert when underlying drops)
    # -----------------------------------------------------------------------
    "BULL_SILVER_X5_AVA_3": {
        "ob_id": "1069606",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL SILVER X5 AVA 3",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    "BULL_SILVER_X3_AVA": {
        "ob_id": "738797",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 3.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL SILVER X3 AVA",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # -----------------------------------------------------------------------
    # Silver SHORT (fish spikes — buy BEAR cert when underlying rises)
    # -----------------------------------------------------------------------
    "BEAR_SILVER_X5_AVA_12": {
        "ob_id": "2286417",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR SILVER X5 AVA 12",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # -----------------------------------------------------------------------
    # Gold LONG (fish dips — buy BULL cert when underlying drops)
    # -----------------------------------------------------------------------
    "BULL_GULD_X5_AVA": {
        "ob_id": "738811",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL GULD X5 AVA",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    "BULL_GULD_X5_AVA_1": {
        "ob_id": "716093",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL GULD X5 AVA 1",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # -----------------------------------------------------------------------
    # Gold SHORT (fish spikes — buy BEAR cert when underlying rises)
    # Only viable X5 is VON (non-AVA, 2.2% spread). X2 AVA as fallback.
    # -----------------------------------------------------------------------
    "BEAR_GULD_X5_VON4": {
        "ob_id": "1047859",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR GULD X5 VON4",
        "issuer": "VON",
        "spread_pct": 2.2,
        "commission_sek": 0,
    },
    "BEAR_GULD_X2_AVA": {
        "ob_id": "738805",
        "api_type": "certificate",
        "underlying": "XAU-USD",
        "direction": "SHORT",
        "leverage": 2.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR GULD X2 AVA",
        "issuer": "AVA",
        "spread_pct": 0.5,
        "commission_sek": 0,
    },
    # -----------------------------------------------------------------------
    # MINI S Silver — overnight-holdable SHORT warrants (no daily reset)
    # Barrier = knockout level. Higher barrier = safer but lower leverage.
    # -----------------------------------------------------------------------
    "MINI_S_SILVER_AVA_409": {
        "ob_id": "2374804",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 10.0,
        "barrier": 80.97,
        "parity": 1,
        "name": "MINI S SILVER AVA 409",
        "issuer": "AVA",
        "spread_pct": 0.3,
        "commission_sek": 0,
    },
    "MINI_S_SILVER_AVA_405": {
        "ob_id": "2367822",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 6.0,
        "barrier": 85.94,
        "parity": 1,
        "name": "MINI S SILVER AVA 405",
        "issuer": "AVA",
        "spread_pct": 0.2,
        "commission_sek": 0,
    },
    "MINI_S_SILVER_AVA_414": {
        "ob_id": "2374783",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 30.0,
        "barrier": 77.50,
        "parity": 1,
        "name": "MINI S SILVER AVA 414",
        "issuer": "AVA",
        "spread_pct": 0.9,
        "commission_sek": 0,
    },
    # -----------------------------------------------------------------------
    # MINI L Silver — overnight-holdable LONG warrants (no daily reset)
    # -----------------------------------------------------------------------
    "MINI_L_SILVER_SG": {
        "ob_id": "2043157",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 1.56,
        "barrier": 32.45,
        "parity": 10,
        "name": "MINI L SILVER SG",
        "issuer": "SG",
        "spread_pct": 2.5,
        "commission_sek": 0,
    },
}

# ---------------------------------------------------------------------------
# Preferred instruments — quick lookup for each ticker+direction combo
# ---------------------------------------------------------------------------
PREFERRED_INSTRUMENTS = {
    ("XAG-USD", "LONG"): "BULL_SILVER_X5_AVA_3",
    ("XAG-USD", "SHORT"): "BEAR_SILVER_X5_AVA_12",
    ("XAU-USD", "LONG"): "BULL_GULD_X5_AVA",
    ("XAU-USD", "SHORT"): "BEAR_GULD_X5_VON4",
}

# Overnight instruments — MINI warrants (no daily reset, can hold overnight)
# Prefer wider barrier (safer) by default. Higher leverage = closer barrier.
OVERNIGHT_INSTRUMENTS = {
    ("XAG-USD", "SHORT"): "MINI_S_SILVER_AVA_405",   # 6x, barrier $85.94 (safest)
    ("XAG-USD", "LONG"): "MINI_L_SILVER_SG",         # 1.56x, barrier $32.45
}
# Alternative overnight SHORT instruments ranked by barrier distance (safest first):
#   MINI_S_SILVER_AVA_405: 6x lev, barrier $85.94 (+14% from $75)
#   MINI_S_SILVER_AVA_409: 10x lev, barrier $80.97 (+8% from $75)
#   MINI_S_SILVER_AVA_414: 30x lev, barrier $77.50 (+3% from $75) — DANGER

# ---------------------------------------------------------------------------
# Master switches
# ---------------------------------------------------------------------------
FISHING_ENABLED = False           # Master on/off — start with OFF
FISHING_DRY_RUN = True            # Log decisions without placing orders
FISHING_ACCOUNT_ID = "1625505"    # Avanza account

# ---------------------------------------------------------------------------
# Session gating
# ---------------------------------------------------------------------------
FISHING_START_HOUR_CET = 8       # Don't fish before 08:15
FISHING_START_MIN_CET = 15
FISHING_LAST_ENTRY_HOUR_CET = 18 # Don't open new positions after 18:55
FISHING_LAST_ENTRY_MIN_CET = 55
FISHING_EOD_CANCEL_HOUR_CET = 21 # Cancel unfilled orders at 21:00
FISHING_EOD_CANCEL_MIN_CET = 0
FISHING_EOD_EXIT_HOUR_CET = 21   # Force-sell positions at 21:00
FISHING_EOD_EXIT_MIN_CET = 0

# ---------------------------------------------------------------------------
# Sizing & limits
# ---------------------------------------------------------------------------
FISHING_BUDGET_SEK = 20_000      # Budget per fishing level
FISHING_MAX_CONCURRENT = 2       # Max concurrent fishing orders per underlying
FISHING_MAX_TOTAL_BUDGET = 60_000 # Max total capital in fishing orders

# ---------------------------------------------------------------------------
# Entry rules
# ---------------------------------------------------------------------------
FISHING_MIN_FILL_PROB = 0.05     # Min fill probability to place order
FISHING_MIN_EV_SEK = 30          # Min expected value in SEK
FISHING_REPRICE_THRESHOLD_PCT = 1.0  # Reprice if spot moves >1%
FISHING_PREFER_AVA = True        # Prefer AVA instruments (no fees)
FISHING_PREFER_LEVERAGE = 5      # Preferred leverage (5x)

# ---------------------------------------------------------------------------
# Exit cascade — take-profit levels
# ---------------------------------------------------------------------------
FISHING_TP_CASCADE = [
    {"underlying_pct": 1.5, "sell_pct": 40, "action": "move_stop_to_breakeven"},
    {"underlying_pct": 2.5, "sell_pct": 40, "action": "trail_stop_1pct"},
    {"underlying_pct": 4.0, "sell_pct": 20, "action": "close"},
]

# ---------------------------------------------------------------------------
# Exit cascade — stop-loss levels
# ---------------------------------------------------------------------------
FISHING_SL_CASCADE = [
    {"underlying_pct": -1.0, "sell_pct": 50, "action": "partial_stop"},
    {"underlying_pct": -2.0, "sell_pct": 100, "action": "full_stop"},
]

# ---------------------------------------------------------------------------
# Time limits
# ---------------------------------------------------------------------------
FISHING_MAX_HOLD_HOURS = 5.0     # Force sell after 5h
FISHING_TIGHTEN_STOP_AFTER_HOURS = 3.0  # Tighten stop to -0.5% after 3h
FISHING_OVERNIGHT_PROTECTION = True     # ALWAYS cancel+sell at session end

# ---------------------------------------------------------------------------
# Direction logic
# ---------------------------------------------------------------------------
FISHING_BULL_WHEN_RSI_BELOW = 45    # Fish BULL (buy dip) when RSI < 45
FISHING_BEAR_WHEN_RSI_ABOVE = 65    # Fish BEAR (buy peak) when RSI > 65
FISHING_NEUTRAL_ZONE = (45, 65)     # Both directions disabled in neutral

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
FISHING_TELEGRAM_ON_PLAN = True     # Send plan via Telegram
FISHING_TELEGRAM_ON_FILL = True     # Alert on fill
FISHING_TELEGRAM_ON_EXIT = True     # Alert on exit
FISHING_TELEGRAM_FALLBACK = True    # Send plan when Avanza unavailable

# ---------------------------------------------------------------------------
# State files
# ---------------------------------------------------------------------------
FISHING_STATE_FILE = "data/fin_fish_state.json"
FISHING_LOG_FILE = "data/fin_fish_log.jsonl"
