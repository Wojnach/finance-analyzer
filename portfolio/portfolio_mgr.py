"""Portfolio state management — load, save, atomic writes, value calculation."""

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("portfolio.portfolio_mgr")

from portfolio.file_utils import atomic_write_json as _atomic_write_json, load_json

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "portfolio_state.json"
BOLD_STATE_FILE = DATA_DIR / "portfolio_state_bold.json"
INITIAL_CASH_SEK = 500_000

_DEFAULT_STATE = {
    "cash_sek": INITIAL_CASH_SEK,
    "holdings": {},
    "transactions": [],
    "initial_value_sek": INITIAL_CASH_SEK,
}


def _validated_state(loaded):
    """Merge loaded state with defaults to ensure all required keys exist."""
    if not loaded or not isinstance(loaded, dict):
        return {**_DEFAULT_STATE, "start_date": datetime.now(timezone.utc).isoformat()}
    result = {**_DEFAULT_STATE, **loaded}
    # Ensure types are correct for critical fields
    if not isinstance(result.get("holdings"), dict):
        result["holdings"] = {}
    if not isinstance(result.get("transactions"), list):
        result["transactions"] = []
    return result


def load_state():
    """Load Patient portfolio state. Returns validated defaults if missing or corrupt."""
    loaded = load_json(str(STATE_FILE), default=None)
    if loaded is None:
        return {**_DEFAULT_STATE, "start_date": datetime.now(timezone.utc).isoformat()}
    return _validated_state(loaded)


def save_state(state):
    _atomic_write_json(STATE_FILE, state)


def load_bold_state():
    """Load Bold portfolio state. Returns validated defaults if missing or corrupt."""
    loaded = load_json(str(BOLD_STATE_FILE), default=None)
    if loaded is None:
        return {**_DEFAULT_STATE, "start_date": datetime.now(timezone.utc).isoformat()}
    return _validated_state(loaded)


def save_bold_state(state):
    """Save Bold portfolio state atomically."""
    _atomic_write_json(BOLD_STATE_FILE, state)


def portfolio_value(state, prices_usd, fx_rate):
    if not isinstance(fx_rate, (int, float)) or fx_rate <= 0:
        logger.warning("portfolio_value: invalid fx_rate=%r, returning cash only", fx_rate)
        return state.get("cash_sek", 0)
    total = state.get("cash_sek", 0)
    for ticker, h in state.get("holdings", {}).items():
        try:
            shares = h.get("shares", 0)
            price = prices_usd.get(ticker)
            if shares > 0 and price is not None:
                total += shares * price * fx_rate
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning("portfolio_value: error calculating %s: %s", ticker, e)
    return total
