"""Portfolio state management — load, save, atomic writes, value calculation."""

import logging
import math
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("portfolio.portfolio_mgr")

from portfolio.file_utils import atomic_write_json as _atomic_write_json
from portfolio.file_utils import load_json

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

# C8: Per-file locks for concurrency safety
_state_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()

_MAX_BACKUPS = 3  # Rolling backup count


def _get_lock(path: Path) -> threading.Lock:
    """Get or create a per-file lock for concurrency safety (C8)."""
    key = str(path)
    with _locks_lock:
        if key not in _state_locks:
            _state_locks[key] = threading.Lock()
        return _state_locks[key]


def _rotate_backups(path: Path):
    """C7: Create rolling .bak backups before overwriting state.

    Maintains up to _MAX_BACKUPS copies: path.bak, path.bak2, path.bak3.
    Only backs up if the file exists and has content.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        # Rotate existing backups: .bak2 → .bak3, .bak → .bak2
        for i in range(_MAX_BACKUPS, 1, -1):
            src = path.with_suffix(f".json.bak{i - 1}" if i > 2 else ".json.bak")
            dst = path.with_suffix(f".json.bak{i}")
            if src.exists():
                shutil.copy2(str(src), str(dst))
        # Current file → .bak
        shutil.copy2(str(path), str(path.with_suffix(".json.bak")))
    except OSError as e:
        logger.warning("Failed to rotate backups for %s: %s", path.name, e)


def _validated_state(loaded):
    """Merge loaded state with defaults to ensure all required keys exist."""
    if not loaded or not isinstance(loaded, dict):
        return {**_DEFAULT_STATE, "start_date": datetime.now(UTC).isoformat()}
    result = {**_DEFAULT_STATE, **loaded}
    # Ensure types are correct for critical fields
    if not isinstance(result.get("holdings"), dict):
        result["holdings"] = {}
    if not isinstance(result.get("transactions"), list):
        result["transactions"] = []
    return result


def _load_state_from(path: Path):
    """Load portfolio state from a specific file.

    C7: On corruption, logs CRITICAL and attempts recovery from backups.
    Returns validated defaults only if file AND all backups are missing/corrupt.
    """
    loaded = load_json(str(path), default=None)
    if loaded is not None:
        return _validated_state(loaded)

    # File is missing or corrupt — check if the file exists (corruption vs missing)
    if path.exists():
        logger.critical(
            "CORRUPT portfolio state file: %s — attempting backup recovery", path.name
        )
        # Try backups in order
        for i in range(1, _MAX_BACKUPS + 1):
            bak = path.with_suffix(f".json.bak{i}" if i > 1 else ".json.bak")
            if bak.exists():
                loaded = load_json(str(bak), default=None)
                if loaded is not None:
                    logger.warning("Recovered %s from backup %s", path.name, bak.name)
                    return _validated_state(loaded)
        logger.critical(
            "ALL backups corrupt/missing for %s — returning fresh defaults", path.name
        )

    return {**_DEFAULT_STATE, "start_date": datetime.now(UTC).isoformat()}


def _save_state_to(path: Path, state):
    """Save state with rolling backup (C7) and lock (C8)."""
    lock = _get_lock(path)
    with lock:
        _rotate_backups(path)
        _atomic_write_json(path, state)


def load_state():
    """Load Patient portfolio state. Returns validated defaults if missing or corrupt."""
    return _load_state_from(STATE_FILE)


def save_state(state):
    """Save Patient portfolio state with backup rotation (C7)."""
    _save_state_to(STATE_FILE, state)


def load_bold_state():
    """Load Bold portfolio state. Returns validated defaults if missing or corrupt."""
    return _load_state_from(BOLD_STATE_FILE)


def save_bold_state(state):
    """Save Bold portfolio state with backup rotation (C7)."""
    _save_state_to(BOLD_STATE_FILE, state)


def update_state(mutate_fn, bold=False):
    """Atomic read-modify-write for portfolio state (C8).

    Holds a lock for the entire read-modify-write cycle to prevent
    concurrent callers from overwriting each other's mutations.

    Args:
        mutate_fn: Callable that receives the current state dict and mutates it.
            The function should modify the dict in-place and optionally return it.
        bold: If True, operates on the Bold portfolio instead of Patient.

    Returns:
        The updated state dict.
    """
    path = BOLD_STATE_FILE if bold else STATE_FILE
    lock = _get_lock(path)
    with lock:
        state = _load_state_from(path)
        result = mutate_fn(state)
        if result is not None:
            state = result
        _rotate_backups(path)
        _atomic_write_json(path, state)
    return state


def portfolio_value(state, prices_usd, fx_rate):
    if not isinstance(fx_rate, (int, float)) or not math.isfinite(fx_rate) or fx_rate <= 0:
        logger.warning("portfolio_value: invalid fx_rate=%r, returning cash only", fx_rate)
        return state.get("cash_sek", 0)
    total = state.get("cash_sek", 0)
    for ticker, h in state.get("holdings", {}).items():
        try:
            shares = h.get("shares", 0)
            price = prices_usd.get(ticker)
            if shares > 0 and price is not None and price > 0:
                total += shares * price * fx_rate
            elif shares > 0 and (price is None or price <= 0):
                logger.warning(
                    "portfolio_value: invalid price for %s: %r (shares=%s)",
                    ticker, price, shares,
                )
        except (TypeError, ValueError, AttributeError) as e:
            logger.warning("portfolio_value: error calculating %s: %s", ticker, e)
    return total
