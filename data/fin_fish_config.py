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
    "BEAR_SILVER_X4_AVA_8": {
        "ob_id": "2246986",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 4.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR SILVER X4 AVA 8",
        "issuer": "AVA",
        "spread_pct": 0.17,
        "commission_sek": 0,
    },
    "BEAR_SILVER_X10_AVA_39": {
        "ob_id": "2340881",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 10.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR SILVER X10 AVA 39",
        "issuer": "AVA",
        "spread_pct": 0.23,
        "commission_sek": 0,
    },
    "BEAR_SILVER_X10_AVA_38": {
        "ob_id": "2323744",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 10.0,
        "barrier": 0,
        "parity": 1,
        "name": "BEAR SILVER X10 AVA 38",
        "issuer": "AVA",
        "spread_pct": 0.92,
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
    "MINI_S_SILVER_AVA_413": {
        "ob_id": "2374796",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 12.5,
        "barrier": 76.97,
        "parity": 1,
        "name": "MINI S SILVER AVA 413",
        "issuer": "AVA",
        "spread_pct": 0.36,
        "commission_sek": 0,
    },
    "MINI_S_SILVER_AVA_412": {
        "ob_id": "2374739",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 10.52,
        "barrier": 77.97,
        "parity": 1,
        "name": "MINI S SILVER AVA 412",
        "issuer": "AVA",
        "spread_pct": 0.31,
        "commission_sek": 0,
    },
    "MINI_S_SILVER_AVA_411": {
        "ob_id": "2374789",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 9.14,
        "barrier": 78.97,
        "parity": 1,
        "name": "MINI S SILVER AVA 411",
        "issuer": "AVA",
        "spread_pct": 0.27,
        "commission_sek": 0,
    },
    "MINI_S_SILVER_AVA_410": {
        "ob_id": "2374817",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 8.12,
        "barrier": 79.97,
        "parity": 1,
        "name": "MINI S SILVER AVA 410",
        "issuer": "AVA",
        "spread_pct": 0.24,
        "commission_sek": 0,
    },
    "MINI_S_SILVER_AVA_407": {
        "ob_id": "2367884",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.57,
        "barrier": 83.95,
        "parity": 1,
        "name": "MINI S SILVER AVA 407",
        "issuer": "AVA",
        "spread_pct": 0.16,
        "commission_sek": 0,
    },
    "MINI_S_SILVER_AVA_401": {
        "ob_id": "2361428",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 3.79,
        "barrier": 89.92,
        "parity": 1,
        "name": "MINI S SILVER AVA 401",
        "issuer": "AVA",
        "spread_pct": 0.11,
        "commission_sek": 0,
    },
    "MINI_S_SILVER_AVA_394": {
        "ob_id": "2345694",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 3.27,
        "barrier": 92.84,
        "parity": 1,
        "name": "MINI S SILVER AVA 394",
        "issuer": "AVA",
        "spread_pct": 0.10,
        "commission_sek": 0,
    },
    "MINI_S_SILVER_AVA_359": {
        "ob_id": "2312392",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 2.44,
        "barrier": 100.19,
        "parity": 1,
        "name": "MINI S SILVER AVA 359",
        "issuer": "AVA",
        "spread_pct": 0.07,
        "commission_sek": 0,
    },
    # -----------------------------------------------------------------------
    # TURBO L Silver — high-leverage LONG with knockout barrier
    # Like MINIs but higher leverage. AVA = zero commission, tight spread.
    # -----------------------------------------------------------------------
    "TURBO_L_SILVER_AVA_481": {
        "ob_id": "2379773",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 7.2,
        "barrier": 65.0,
        "parity": 1,
        "name": "TURBO L SILVER AVA 481",
        "issuer": "AVA",
        "spread_pct": 0.2,
        "commission_sek": 0,
    },
    "TURBO_L_SILVER_AVA_482": {
        "ob_id": "2379784",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 7.9,
        "barrier": 66.0,
        "parity": 1,
        "name": "TURBO L SILVER AVA 482",
        "issuer": "AVA",
        "spread_pct": 0.2,
        "commission_sek": 0,
    },
    "TURBO_L_SILVER_AVA_490": {
        "ob_id": "2389097",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 11.02,
        "barrier": 67.04,
        "parity": 1,
        "name": "TURBO L SILVER AVA 490",
        "issuer": "AVA",
        "spread_pct": 0.32,
        "commission_sek": 0,
    },
    "TURBO_L_SILVER_AVA_491": {
        "ob_id": "2389098",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 9.8,
        "barrier": 68.0,
        "parity": 1,
        "name": "TURBO L SILVER AVA 491",
        "issuer": "AVA",
        "spread_pct": 0.37,
        "commission_sek": 0,
    },
    "TURBO_L_SILVER_AVA_492": {
        "ob_id": "2390438",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 15.25,
        "barrier": 69.04,
        "parity": 1,
        "name": "TURBO L SILVER AVA 492",
        "issuer": "AVA",
        "spread_pct": 0.44,
        "commission_sek": 0,
    },
    "TURBO_L_SILVER_AVA_355": {
        "ob_id": "2246993",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 5.97,
        "barrier": 61.20,
        "parity": 1,
        "name": "TURBO L SILVER AVA 355",
        "issuer": "AVA",
        "spread_pct": 0.17,
        "commission_sek": 0,
    },
    # -----------------------------------------------------------------------
    # TURBO S Silver — high-leverage SHORT with knockout barrier
    # -----------------------------------------------------------------------
    "TURBO_S_SILVER_AVA_522": {
        "ob_id": "2345696",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 3.56,
        "barrier": 92.73,
        "parity": 1,
        "name": "TURBO S SILVER AVA 522",
        "issuer": "AVA",
        "spread_pct": 0.10,
        "commission_sek": 0,
    },
    "TURBO_S_SILVER_AVA_539": {
        "ob_id": "2361437",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.85,
        "barrier": 84.82,
        "parity": 1,
        "name": "TURBO S SILVER AVA 539",
        "issuer": "AVA",
        "spread_pct": 0.17,
        "commission_sek": 0,
    },
    "TURBO_S_SILVER_AVA_541": {
        "ob_id": "2367831",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 6.98,
        "barrier": 82.85,
        "parity": 1,
        "name": "TURBO S SILVER AVA 541",
        "issuer": "AVA",
        "spread_pct": 0.20,
        "commission_sek": 0,
    },
    "TURBO_S_SILVER_AVA_544": {
        "ob_id": "2370161",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 7.72,
        "barrier": 81.86,
        "parity": 1,
        "name": "TURBO S SILVER AVA 544",
        "issuer": "AVA",
        "spread_pct": 0.23,
        "commission_sek": 0,
    },
    "TURBO_S_SILVER_AVA_545": {
        "ob_id": "2374841",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 12.94,
        "barrier": 77.88,
        "parity": 1,
        "name": "TURBO S SILVER AVA 545",
        "issuer": "AVA",
        "spread_pct": 0.38,
        "commission_sek": 0,
    },
    "TURBO_S_SILVER_AVA_546": {
        "ob_id": "2374815",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 15.49,
        "barrier": 76.88,
        "parity": 1,
        "name": "TURBO S SILVER AVA 546",
        "issuer": "AVA",
        "spread_pct": 0.23,
        "commission_sek": 0,
    },
    "TURBO_S_SILVER_AVA_553": {
        "ob_id": "2383713",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 9.57,
        "barrier": 79.91,
        "parity": 1,
        "name": "TURBO S SILVER AVA 553",
        "issuer": "AVA",
        "spread_pct": 0.28,
        "commission_sek": 0,
    },
    "TURBO_S_SILVER_AVA_554": {
        "ob_id": "2383705",
        "api_type": "warrant",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 15.2,
        "barrier": 79.0,
        "parity": 1,
        "name": "TURBO S SILVER AVA 554",
        "issuer": "AVA",
        "spread_pct": 0.33,
        "commission_sek": 0,
    },
    # -----------------------------------------------------------------------
    # Better BULL cert — BULL SILVER X5 AVA 4 has 4x more volume, 0.2% spread
    # -----------------------------------------------------------------------
    "BULL_SILVER_X5_AVA_4": {
        "ob_id": "1650161",
        "api_type": "certificate",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
        "parity": 1,
        "name": "BULL SILVER X5 AVA 4",
        "issuer": "AVA",
        "spread_pct": 0.2,
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
    ("XAG-USD", "LONG"): "BULL_SILVER_X5_AVA_4",  # 0.2% spread, 1.4M vol (better than AVA_3)
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
