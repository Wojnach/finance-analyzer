"""Single source of truth for all ticker lists, source mappings, and symbol constants.

Every module that needs ticker definitions should import from here instead
of maintaining its own copy.
"""

# ── Tier 1: Full signals (25 signals, 7 timeframes) ──────────────────────

SYMBOLS = {
    # Crypto (Binance spot)
    "BTC-USD": {"binance": "BTCUSDT"},
    "ETH-USD": {"binance": "ETHUSDT"},
    # Metals (Binance futures)
    "XAU-USD": {"binance_fapi": "XAUUSDT"},
    "XAG-USD": {"binance_fapi": "XAGUSDT"},
    # US Equities (Alpaca IEX)
    "MSTR": {"alpaca": "MSTR"},
    "PLTR": {"alpaca": "PLTR"},
    "NVDA": {"alpaca": "NVDA"},
    "AMD": {"alpaca": "AMD"},
    "BABA": {"alpaca": "BABA"},
    "GOOGL": {"alpaca": "GOOGL"},
    "AMZN": {"alpaca": "AMZN"},
    "AAPL": {"alpaca": "AAPL"},
    "AVGO": {"alpaca": "AVGO"},
    "AI": {"alpaca": "AI"},
    "GRRR": {"alpaca": "GRRR"},
    "IONQ": {"alpaca": "IONQ"},
    "MRVL": {"alpaca": "MRVL"},
    "META": {"alpaca": "META"},
    "MU": {"alpaca": "MU"},
    "PONY": {"alpaca": "PONY"},
    "RXRX": {"alpaca": "RXRX"},
    "SOUN": {"alpaca": "SOUN"},
    "SMCI": {"alpaca": "SMCI"},
    "TSM": {"alpaca": "TSM"},
    "TTWO": {"alpaca": "TTWO"},
    "TEM": {"alpaca": "TEM"},
    "UPST": {"alpaca": "UPST"},
    "VERI": {"alpaca": "VERI"},
    "VRT": {"alpaca": "VRT"},
    "QQQ": {"alpaca": "QQQ"},
    "LMT": {"alpaca": "LMT"},
}

# ── Asset-class subsets ───────────────────────────────────────────────────

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}
METALS_SYMBOLS = {"XAU-USD", "XAG-USD"}
STOCK_SYMBOLS = {
    "MSTR", "PLTR", "NVDA", "AMD", "BABA", "GOOGL", "AMZN", "AAPL",
    "AVGO", "AI", "GRRR", "IONQ", "MRVL", "META", "MU", "PONY",
    "RXRX", "SOUN", "SMCI", "TSM", "TTWO", "TEM", "UPST", "VERI",
    "VRT", "QQQ", "LMT",
}

# All known tickers (union of all subsets)
ALL_TICKERS = CRYPTO_SYMBOLS | METALS_SYMBOLS | STOCK_SYMBOLS

# ── Binance symbol mappings (used by outcome_tracker, macro_context) ─────

BINANCE_SPOT_MAP = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
BINANCE_FAPI_MAP = {"XAU-USD": "XAUUSDT", "XAG-USD": "XAGUSDT"}
BINANCE_MAP = {**BINANCE_SPOT_MAP, **BINANCE_FAPI_MAP}

# ── Ticker → (source_type, symbol) mapping (used by macro_context) ───────

TICKER_SOURCE_MAP = {
    "BTC-USD": ("binance", "BTCUSDT"),
    "ETH-USD": ("binance", "ETHUSDT"),
    "XAU-USD": ("binance_fapi", "XAUUSDT"),
    "XAG-USD": ("binance_fapi", "XAGUSDT"),
    "MSTR": ("alpaca", "MSTR"),
    "PLTR": ("alpaca", "PLTR"),
    "NVDA": ("alpaca", "NVDA"),
    "AMD": ("alpaca", "AMD"),
    "BABA": ("alpaca", "BABA"),
    "GOOGL": ("alpaca", "GOOGL"),
    "AMZN": ("alpaca", "AMZN"),
    "AAPL": ("alpaca", "AAPL"),
    "AVGO": ("alpaca", "AVGO"),
    "AI": ("alpaca", "AI"),
    "GRRR": ("alpaca", "GRRR"),
    "IONQ": ("alpaca", "IONQ"),
    "MRVL": ("alpaca", "MRVL"),
    "META": ("alpaca", "META"),
    "MU": ("alpaca", "MU"),
    "PONY": ("alpaca", "PONY"),
    "RXRX": ("alpaca", "RXRX"),
    "SOUN": ("alpaca", "SOUN"),
    "SMCI": ("alpaca", "SMCI"),
    "TSM": ("alpaca", "TSM"),
    "TTWO": ("alpaca", "TTWO"),
    "TEM": ("alpaca", "TEM"),
    "UPST": ("alpaca", "UPST"),
    "VERI": ("alpaca", "VERI"),
    "VRT": ("alpaca", "VRT"),
    "QQQ": ("alpaca", "QQQ"),
    "LMT": ("alpaca", "LMT"),
}

# ── Yahoo Finance symbol mapping (used by outcome_tracker) ───────────────

YF_MAP = {
    "MSTR": "MSTR", "PLTR": "PLTR", "NVDA": "NVDA",
    "AMD": "AMD", "BABA": "BABA", "GOOGL": "GOOGL", "AMZN": "AMZN",
    "AAPL": "AAPL", "AVGO": "AVGO", "AI": "AI", "GRRR": "GRRR",
    "IONQ": "IONQ", "MRVL": "MRVL", "META": "META", "MU": "MU",
    "PONY": "PONY", "RXRX": "RXRX", "SOUN": "SOUN", "SMCI": "SMCI",
    "TSM": "TSM", "TTWO": "TTWO", "TEM": "TEM", "UPST": "UPST",
    "VERI": "VERI", "VRT": "VRT", "QQQ": "QQQ", "LMT": "LMT",
}

# ── Signal names (used by outcome_tracker, accuracy_stats) ───────────────

SIGNAL_NAMES = [
    "rsi",
    "macd",
    "ema",
    "bb",
    "fear_greed",
    "sentiment",
    "ministral",
    "ml",
    "funding",
    "volume",
    "custom_lora",
    # Enhanced composite signals
    "trend",
    "momentum",
    "volume_flow",
    "volatility_sig",
    "candlestick",
    "structure",
    "fibonacci",
    "smart_money",
    "oscillators",
    "heikin_ashi",
    "mean_reversion",
    "calendar",
    "macro_regime",
    "momentum_factors",
]
