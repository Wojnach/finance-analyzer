"""Module-level state reset helpers for xdist-safe testing.

xdist distributes test files across workers. Each worker's Python process
imports the same production modules, and module-level mutable state (globals)
persists across test files sharded to the same worker. If test A mutates
``agent_invocation._agent_proc`` and test B expects it to be None, B fails
when sharded to the same worker — but passes in isolation.

These helpers reset each module's state to its default values. Called from
autouse fixtures in conftest.py (global) and in individual test files for
medium-risk modules.

See: docs/IMPROVEMENT_BACKLOG.md TEST-HYGIENE-1 for the full problem description.
"""

import logging

logger = logging.getLogger(__name__)


def reset_agent_invocation():
    """Reset agent_invocation module-level state to defaults."""
    try:
        import portfolio.agent_invocation as ai
        ai._agent_proc = None
        ai._agent_log = None
        ai._agent_log_start_offset = 0
        ai._agent_start = 0
        ai._agent_start_wall = 0.0
        ai._agent_timeout = 900
        ai._agent_tier = None
        ai._agent_reasons = None
        ai._journal_ts_before = None
        ai._telegram_ts_before = None
        # Don't reset _consecutive_stack_overflows — it's loaded from disk
        # and resetting it could mask a real stack overflow in tests.
    except ImportError:
        pass


def reset_signal_engine():
    """Reset signal_engine module-level caches and state."""
    try:
        import portfolio.signal_engine as se
        with se._adx_lock:
            se._adx_cache.clear()
        with se._last_signal_lock:
            se._last_signal_per_ticker.clear()
        with se._phase_log_lock:
            se._phase_log_per_ticker.clear()
        with se._sentiment_lock:
            se._prev_sentiment.clear()
            se._prev_sentiment_loaded = False
            se._sentiment_dirty = False
    except ImportError:
        pass


def reset_shared_state():
    """Reset shared_state module-level caches."""
    try:
        import portfolio.shared_state as ss
        with ss._cache_lock:
            ss._tool_cache.clear()
            ss._loading_keys.clear()
        ss._run_cycle_id = 0
        ss._regime_cache.clear()
        ss._regime_cache_cycle = 0
        ss._full_llm_cycle_count = 0
        ss._current_market_state = "open"
        ss._newsapi_daily_count = 0
    except ImportError:
        pass


def reset_forecast_signal():
    """Reset forecast signal module-level state."""
    try:
        import portfolio.signals.forecast as fc
        fc._FORECAST_MODELS_DISABLED = False
        fc._kronos_tripped_until = 0
        fc._chronos_tripped_until = 0
        if hasattr(fc, '_predictions_dedup_cache'):
            fc._predictions_dedup_cache.clear()
    except ImportError:
        pass


def reset_logging_config():
    """Reset logging_config state."""
    try:
        import portfolio.logging_config as lc
        lc._configured = False
    except ImportError:
        pass


def reset_api_utils():
    """Reset api_utils config cache."""
    try:
        import portfolio.api_utils as au
        au._config_cache = None
        if hasattr(au, '_config_mtime'):
            au._config_mtime = None
    except ImportError:
        pass


def reset_trigger():
    """Reset trigger module state."""
    try:
        import portfolio.trigger as tr
        tr._startup_grace_active = True
    except ImportError:
        pass


def reset_all_high_risk():
    """Reset all HIGH-risk module state. Call from global autouse fixture."""
    reset_agent_invocation()
    reset_signal_engine()
    reset_shared_state()


def reset_all():
    """Reset ALL module state (high + medium + low risk)."""
    reset_all_high_risk()
    reset_forecast_signal()
    reset_logging_config()
    reset_api_utils()
    reset_trigger()
