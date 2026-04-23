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
    # US Equities (Alpaca IEX) — MSTR kept as BTC NAV-premium reference for metals_loop
    "MSTR": {"alpaca": "MSTR"},
    # Removed Mar 15: AMD, GOOGL, AMZN, AAPL, AVGO, META, SOUN, LMT
    # Removed Apr 09: PLTR, NVDA, MU, SMCI, TSM, TTWO, VRT
    #   Reduces main loop load to stay under 60s cadence. Cycle p50 was 143s with
    #   12 tickers — dropping to 5 is expected to bring p50 under target. MSTR retained
    #   because data/metals_loop.py uses it for BTC NAV-premium tracking.
}

# ── Asset-class subsets ───────────────────────────────────────────────────

CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}
METALS_SYMBOLS = {"XAU-USD", "XAG-USD"}
STOCK_SYMBOLS = {"MSTR"}

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
DISABLED_SIGNALS = {
    "ml",               # 41.7% accuracy (1714 sam) — worse than coin flip
    # "cot_positioning" re-enabled 2026-04-13 for shadow validation (was
    # force-HOLD pending live validation, 0 samples). COT is a weekly signal
    # (CFTC Friday release) — expected to contribute mostly at 3d/5d horizons
    # where the system already has edge (XAG 5d consensus 61.2%). The
    # existing accuracy gate in signal_engine.py auto-disables any signal
    # below 45% accuracy once 30+ samples accumulate, so re-enabling is
    # self-correcting.
    "futures_basis",    # 0 accuracy samples — pending live validation
    "hurst_regime",     # pending live validation (added 2026-04-11)
    "shannon_entropy",  # pending live validation (added 2026-04-12)
    "vix_term_structure",  # pending live validation (added 2026-04-13)
    "gold_real_yield_paradox",  # pending live validation (added 2026-04-14)
    "cross_asset_tsmom",  # pending live validation (added 2026-04-15)
    "copper_gold_ratio",  # pending live validation (added 2026-04-17)
    "statistical_jump_regime",  # pending live validation (added 2026-04-18)
    "network_momentum",  # pending live validation (added 2026-04-19)
    "ovx_metals_spillover",  # pending live validation (added 2026-04-20)
    "xtrend_equity_spillover",  # pending live validation (added 2026-04-21)
    "complexity_gap_regime",  # pending live validation (added 2026-04-22)
    "realized_skewness",  # pending live validation (added 2026-04-23)
    "econ_calendar",    # BUG-218: structurally SELL-only — all 4 sub-signals can only produce
                        # SELL or HOLD, never BUY. Permanent SELL-biased voter in consensus.
                        # Force-HOLD until BUY capability is added (needs research into
                        # which economic events are bullish).
    "orderbook_flow",   # 2026-04-11: 51.1% accuracy (360 sam), 93.3% activation rate,
                        # no recent data. Pure noise in every consensus decision.
                        # Re-evaluate after 2 weeks of accuracy data collection.
    # "forecast" RE-ENABLED 2026-04-21. The 36-39% accuracy measured on 2026-04-12
    # was polluted by Kronos voting 100% HOLD in shadow mode — Kronos occupied 3 of 6
    # slots in _health_weighted_vote whenever its subprocess succeeded, dragging every
    # composite vote toward HOLD regardless of Chronos's verdict. With Kronos retired
    # in portfolio/signals/forecast.py (same PR), the composite is now Chronos-only.
    # Chronos effective accuracy: 1h=45.4%, 24h=52.4% (4d ago). The 47% tiered
    # accuracy gate will force-HOLD 1h while letting 24h contribute. Forecast stayed
    # in this set for 10 days, which ALSO silenced forecast_predictions.jsonl and
    # forecast_health.jsonl because signal_engine.py skips disabled signals before
    # invocation — so we lost all shadow/health visibility while the signal was off.
    # Re-enabling restores both the signal and the logging. If accuracy degrades
    # again post-Kronos-retire, move into REGIME_GATED_SIGNALS (24h-only) rather
    # than re-disabling blindly.
    "oscillators",      # 2026-04-14: below 45% on ALL tickers at 1d (BTC 35.8%, ETH 36.3%,
                        # XAG 34.9%, XAU 40.2%, MSTR 42.6%; 5065 total sam). Also weak at
                        # 3h (34-45% per ticker). Regime-gated in ranging but noise everywhere.
}
# 2026-04-11 research session changes:
# - orderbook_flow DISABLED: 93.3% active, 51.1% accuracy, 0 recent data. Noise.
# - credit_spread_risk ENABLED: 66.9% accuracy (257 sam), BUY 80.3%. Directional
#   gate at 40% will auto-gate SELL (49.1%) while allowing strong BUY votes.
# - crypto_macro ENABLED: 56.5% accuracy (1273 sam). BUY-biased (93%) so bias
#   penalty (0.5x) applies. Provides crypto-specific on-chain edge.
# funding: removed from DISABLED — 74.2% at 3h (535 samples) but 29.9% at 1d.
# Horizon-gated via REGIME_GATED_SIGNALS to only vote at 3h/4h.

# Signals that require local GPU inference.
# Skipped for US stocks outside market hours to save GPU resources.
# claude_fundamental excluded — uses remote API, has its own market-hours gate.
GPU_SIGNALS = frozenset({"ministral", "qwen3", "forecast"})

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
    "qwen3",
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
    "crypto_macro",
    "orderbook_flow",
    "metals_cross_asset",
    "dxy_cross_asset",
    "cot_positioning",
    "credit_spread_risk",
    "onchain",
    "futures_basis",
    "hurst_regime",
    "shannon_entropy",
    "vix_term_structure",
    "gold_real_yield_paradox",
    "cross_asset_tsmom",
    "copper_gold_ratio",
    "statistical_jump_regime",
    "network_momentum",
    "ovx_metals_spillover",
    "xtrend_equity_spillover",
    "complexity_gap_regime",
    "realized_skewness",
]
