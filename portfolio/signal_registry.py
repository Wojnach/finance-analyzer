"""Signal registry — plugin system for modular signal management.

Signals register themselves via @register_signal decorator. signal_engine.py
discovers all signals from the registry instead of hardcoded lists.
"""
import importlib
import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger("portfolio.signal_registry")

# Registry storage
_CORE_SIGNALS: Dict[str, dict] = {}
_ENHANCED_SIGNALS: Dict[str, dict] = {}


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
                      requires_context: bool = False):
    """Programmatically register an enhanced signal module."""
    _ENHANCED_SIGNALS[name] = {
        "name": name,
        "type": "enhanced",
        "module_path": module_path,
        "func_name": func_name,
        "requires_macro": requires_macro,
        "requires_context": requires_context,
        "func": None,  # lazy-loaded
    }


def get_enhanced_signals() -> Dict[str, dict]:
    """Return all registered enhanced signals."""
    return dict(_ENHANCED_SIGNALS)


def get_signal_names() -> list:
    """Return all signal names (core + enhanced) in order."""
    core = list(_CORE_SIGNALS.keys())
    enhanced = list(_ENHANCED_SIGNALS.keys())
    return core + enhanced


def load_signal_func(entry: dict) -> Optional[Callable]:
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
        ("mean_reversion", "portfolio.signals.mean_reversion", "compute_mean_reversion_signal"),
        ("calendar", "portfolio.signals.calendar_seasonal", "compute_calendar_signal"),
        ("momentum_factors", "portfolio.signals.momentum_factors", "compute_momentum_factors_signal"),
    ]
    for name, mod_path, func_name in defaults:
        register_enhanced(name, mod_path, func_name)
    # macro_regime is special — requires_macro=True
    register_enhanced("macro_regime", "portfolio.signals.macro_regime",
                      "compute_macro_regime_signal", requires_macro=True)
    # news_event and econ_calendar require context (ticker, config)
    register_enhanced("news_event", "portfolio.signals.news_event",
                      "compute_news_event_signal", requires_context=True)
    register_enhanced("econ_calendar", "portfolio.signals.econ_calendar",
                      "compute_econ_calendar_signal", requires_context=True)


_register_defaults()
