"""Single source of truth for all ticker lists, source mappings, and symbol constants.

Every module that needs ticker definitions should import from here instead
of maintaining its own copy.
"""

# ── Tier 1: Full signals (30 signals, 7 timeframes) ──────────────────────

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
    "GRRR": {"alpaca": "GRRR"},
    "IONQ": {"alpaca": "IONQ"},
    "META": {"alpaca": "META"},
    "MU": {"alpaca": "MU"},
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
    "AVGO", "GRRR", "IONQ", "META", "MU",
    "SOUN", "SMCI", "TSM", "TTWO", "TEM", "UPST", "VERI",
    "VRT", "QQQ", "LMT",
}

# All known tickers (union of all subsets)
ALL_TICKERS = CRYPTO_SYMBOLS | METALS_SYMBOLS | STOCK_SYMBOLS

# ── Derived mappings (all from SYMBOLS — single source of truth) ─────────

BINANCE_SPOT_MAP = {
    t: sym for t, src in SYMBOLS.items()
    for k, sym in src.items() if k == "binance"
}
BINANCE_FAPI_MAP = {
    t: sym for t, src in SYMBOLS.items()
    for k, sym in src.items() if k == "binance_fapi"
}
BINANCE_MAP = {**BINANCE_SPOT_MAP, **BINANCE_FAPI_MAP}

# Ticker -> (source_type, symbol) mapping (used by macro_context)
TICKER_SOURCE_MAP = {
    t: next(iter(src.items())) for t, src in SYMBOLS.items()
}

# Yahoo Finance symbol mapping — stock tickers map to themselves
YF_MAP = {t: t for t in STOCK_SYMBOLS}

# ── Signal names (used by outcome_tracker, accuracy_stats) ───────────────
# Canonical source is portfolio.signal_registry.get_signal_names().
# This static list is kept for backward compatibility with modules that
# import SIGNAL_NAMES directly (outcome_tracker, accuracy_stats).

# Signals that are force-HOLD (disabled due to poor accuracy).
# Kept in SIGNAL_NAMES for historical tracking but excluded from active reports.
DISABLED_SIGNALS = {"ml", "funding"}

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
    # custom_lora removed — disabled signal, was polluting accuracy stats
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
    "news_event",
    "econ_calendar",
    "forecast",
    "claude_fundamental",
    "futures_flow",
]
