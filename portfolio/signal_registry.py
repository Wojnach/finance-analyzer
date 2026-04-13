"""Signal registry — plugin system for modular signal management.

Signals register themselves via @register_signal decorator. signal_engine.py
discovers all signals from the registry instead of hardcoded lists.
"""
import importlib
import logging
from collections.abc import Callable

logger = logging.getLogger("portfolio.signal_registry")

# Registry storage
_CORE_SIGNALS: dict[str, dict] = {}
_ENHANCED_SIGNALS: dict[str, dict] = {}


def register_signal(name: str, signal_type: str = "enhanced",
                    module_path: str = None, func_name: str = None,
                    requires_macro: bool = False):
    """Register a signal in the global registry.

    Can be used as a decorator on compute functions, or called directly
    to register signals programmatically.

    Args:
        name: Signal name (e.g., "trend", "rsi")
        signal_type: "core" or "enhanced"
        module_path: Full module path (e.g., "portfolio.signals.trend")
        func_name: Function name to call (e.g., "compute_trend_signal")
        requires_macro: Whether this signal needs macro context (only macro_regime)
    """
    def decorator(func):
        entry = {
            "name": name,
            "type": signal_type,
            "module_path": module_path or func.__module__,
            "func_name": func_name or func.__name__,
            "requires_macro": requires_macro,
            "func": func,
        }
        if signal_type == "core":
            _CORE_SIGNALS[name] = entry
        else:
            _ENHANCED_SIGNALS[name] = entry
        return func
    return decorator


def register_enhanced(name: str, module_path: str, func_name: str,
                      requires_macro: bool = False,
                      requires_context: bool = False,
                      max_confidence: float = 1.0):
    """Programmatically register an enhanced signal module."""
    _ENHANCED_SIGNALS[name] = {
        "name": name,
        "type": "enhanced",
        "module_path": module_path,
        "func_name": func_name,
        "requires_macro": requires_macro,
        "requires_context": requires_context,
        "max_confidence": max_confidence,
        "func": None,  # lazy-loaded
    }


def get_enhanced_signals() -> dict[str, dict]:
    """Return all registered enhanced signals."""
    return dict(_ENHANCED_SIGNALS)


def get_signal_names() -> list:
    """Return all signal names (core + enhanced) in order."""
    core = list(_CORE_SIGNALS.keys())
    enhanced = list(_ENHANCED_SIGNALS.keys())
    return core + enhanced


def load_signal_func(entry: dict) -> Callable | None:
    """Lazy-load and cache the compute function for a signal."""
    if entry.get("func") is not None:
        return entry["func"]
    try:
        mod = importlib.import_module(entry["module_path"])
        func = getattr(mod, entry["func_name"])
        entry["func"] = func
        return func
    except Exception as e:
        logger.warning("Failed to load signal %s: %s", entry['name'], e)
        return None


# Register all enhanced signals (called at import time)
def _register_defaults():
    """Register the default set of enhanced signal modules."""
    defaults = [
        ("trend", "portfolio.signals.trend", "compute_trend_signal"),
        ("momentum", "portfolio.signals.momentum", "compute_momentum_signal"),
        ("volume_flow", "portfolio.signals.volume_flow", "compute_volume_flow_signal"),
        ("volatility_sig", "portfolio.signals.volatility", "compute_volatility_signal"),
        ("candlestick", "portfolio.signals.candlestick", "compute_candlestick_signal"),
        ("structure", "portfolio.signals.structure", "compute_structure_signal"),
        ("fibonacci", "portfolio.signals.fibonacci", "compute_fibonacci_signal"),
        ("smart_money", "portfolio.signals.smart_money", "compute_smart_money_signal"),
        ("oscillators", "portfolio.signals.oscillators", "compute_oscillator_signal"),
        ("heikin_ashi", "portfolio.signals.heikin_ashi", "compute_heikin_ashi_signal"),
        ("calendar", "portfolio.signals.calendar_seasonal", "compute_calendar_signal"),
    ]
    for name, mod_path, func_name in defaults:
        register_enhanced(name, mod_path, func_name)
    # mean_reversion and momentum_factors require context for seasonality detrending
    register_enhanced("mean_reversion", "portfolio.signals.mean_reversion",
                      "compute_mean_reversion_signal", requires_context=True)
    register_enhanced("momentum_factors", "portfolio.signals.momentum_factors",
                      "compute_momentum_factors_signal", requires_context=True)
    # macro_regime is special — requires_macro=True
    register_enhanced("macro_regime", "portfolio.signals.macro_regime",
                      "compute_macro_regime_signal", requires_macro=True)
    # news_event and econ_calendar require context (ticker, config); capped at 0.7
    register_enhanced("news_event", "portfolio.signals.news_event",
                      "compute_news_event_signal", requires_context=True, max_confidence=0.7)
    register_enhanced("econ_calendar", "portfolio.signals.econ_calendar",
                      "compute_econ_calendar_signal", requires_context=True, max_confidence=0.7)
    # forecast signal — Kronos + Chronos price direction prediction; capped at 0.7
    register_enhanced("forecast", "portfolio.signals.forecast",
                      "compute_forecast_signal", requires_context=True, max_confidence=0.7)
    # Claude fundamental — three-tier LLM cascade; capped at 0.7
    register_enhanced("claude_fundamental", "portfolio.signals.claude_fundamental",
                      "compute_claude_fundamental_signal", requires_context=True, max_confidence=0.7)
    # Futures flow — OI, LS ratios, funding history (crypto only); capped at 0.7
    register_enhanced("futures_flow", "portfolio.signals.futures_flow",
                      "compute_futures_flow_signal", requires_context=True, max_confidence=0.7)
    # Crypto macro — options max pain, gold-BTC rotation, exchange reserves (crypto only); capped at 0.7
    register_enhanced("crypto_macro", "portfolio.signals.crypto_macro",
                      "compute_crypto_macro_signal", requires_context=True, max_confidence=0.7)
    # Orderbook flow — microstructure metrics (metals + crypto); capped at 0.7
    register_enhanced("orderbook_flow", "portfolio.signals.orderbook_flow",
                      "compute_orderbook_flow_signal", requires_context=True, max_confidence=0.7)
    # Metals cross-asset — copper, GVZ, G/S ratio, SPY, oil (metals only); capped at 0.7
    register_enhanced("metals_cross_asset", "portfolio.signals.metals_cross_asset",
                      "compute_metals_cross_asset_signal", requires_context=True, max_confidence=0.7)
    # DXY cross-asset — intraday USD index inverse correlation (metals only); capped at 0.8
    # 2026-04-13: added standalone signal to capture DXY R²~0.6 vs silver at
    # 1-3h horizon. Complements macro_regime's daily DXY sub-indicator.
    register_enhanced("dxy_cross_asset", "portfolio.signals.dxy_cross_asset",
                      "compute_dxy_cross_asset_signal", requires_context=True, max_confidence=0.8)
    # COT positioning — CFTC speculative/commercial positioning, contrarian (metals only); capped at 0.7
    register_enhanced("cot_positioning", "portfolio.signals.cot_positioning",
                      "compute_cot_positioning_signal", requires_context=True, max_confidence=0.7)
    # Credit spread risk — HY OAS from FRED as cross-asset risk appetite gauge; capped at 0.7
    register_enhanced("credit_spread_risk", "portfolio.signals.credit_spread",
                      "compute_credit_spread_signal", requires_context=True, max_confidence=0.7)
    # Futures basis regime — mark-index spread, contango/backwardation detection; capped at 0.7
    register_enhanced("futures_basis", "portfolio.signals.futures_basis",
                      "compute_futures_basis_signal", requires_context=True, max_confidence=0.7)
    # Hurst regime detector — R/S analysis for trending/MR/random-walk classification
    register_enhanced("hurst_regime", "portfolio.signals.hurst_regime",
                      "compute_hurst_regime_signal", requires_context=True)
    # Shannon entropy — market noise/predictability filter; low entropy = trending
    register_enhanced("shannon_entropy", "portfolio.signals.shannon_entropy",
                      "compute_shannon_entropy_signal")


_register_defaults()
