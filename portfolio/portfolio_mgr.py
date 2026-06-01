"""Portfolio state management — load, save, atomic writes, value calculation."""

import hashlib
import logging
import math
import shutil
import threading
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("portfolio.portfolio_mgr")

from portfolio.file_utils import atomic_append_jsonl
from portfolio.file_utils import atomic_write_json as _atomic_write_json
from portfolio.file_utils import load_json

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "portfolio_state.json"
BOLD_STATE_FILE = DATA_DIR / "portfolio_state_bold.json"
# Canonical surfacing path for failures the loop can't self-heal. Kept as a
# module constant (not hard-coded at the call site) so tests can patch it to
# tmp_path under xdist without touching the live journal.
CRITICAL_ERRORS_LOG = DATA_DIR / "critical_errors.jsonl"
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


def _quarantine_corrupt_state(path: Path, corrupt_bytes: bytes) -> None:
    """Preserve a corrupt state file + surface it before fresh defaults overwrite it.

    Added 2026-06-01 after a hand-edit left ``portfolio_state.json`` unparseable
    with NO ``.bak`` on disk: ``_load_state_from`` would have returned fresh
    defaults and the next ``save_state`` would have silently wiped the entire
    portfolio (only a ``logger.critical`` line, no journal entry, no alert). This
    makes that path loud + recoverable.

    Design (see docs/PLAN.md premortem):
    - ``corrupt_bytes`` is captured by the caller the instant corruption is
      detected, BEFORE the backup-recovery loop, so a concurrent lockless
      ``load_state`` reader racing an ``update_state`` writer can't substitute
      fresh-default bytes (premortem #1).
    - Quarantine filename is content-addressed (``<name>.corrupt-<sha8>``) and the
      whole side-effect block is gated on it not already existing, so the corrupt
      branch firing every 60 s cycle quarantines + journals EXACTLY ONCE per
      unique corruption instead of ~1440×/day (premortem #2/#3).
    - No Telegram here: the journal entry is the durable surface and a synchronous
      network send inside the (sometimes lock-held) read path is a stall vector
      and pulls in the worktree-absent config.json symlink (premortem #8).
    - Never raises — a failure to preserve evidence must not crash the read path.
    """
    try:
        if not corrupt_bytes:
            return
        digest = hashlib.sha256(corrupt_bytes).hexdigest()[:8]
        qpath = path.with_name(f"{path.name}.corrupt-{digest}")
        if qpath.exists():
            return  # this exact corruption already preserved + journaled — idempotent
        qpath.write_bytes(corrupt_bytes)
        atomic_append_jsonl(str(CRITICAL_ERRORS_LOG), {
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": "critical",
            "category": "portfolio_state_corrupt",
            "caller": "portfolio.portfolio_mgr",
            "message": (
                f"{path.name} is unparseable and no backup recovered. Preserved "
                f"to {qpath.name} and returned fresh defaults so the loop keeps "
                f"running — RESTORE from {qpath.name} or a .bak before the next "
                f"save overwrites state."
            ),
            "context": {
                "path": str(path),
                "quarantine": str(qpath),
                "sha8": digest,
                "bytes": len(corrupt_bytes),
            },
        })
        logger.critical(
            "Quarantined corrupt %s -> %s (returned fresh defaults)",
            path.name, qpath.name,
        )
    except Exception:  # noqa: BLE001 — evidence-preservation is best-effort
        logger.exception("quarantine of corrupt %s failed", path.name)


def _load_state_from(path: Path):
    """Load portfolio state from a specific file.

    C7: On corruption, logs CRITICAL and attempts recovery from backups.
    Returns validated defaults only if file AND all backups are missing/corrupt —
    and in that case quarantines the corrupt bytes + journals a critical first so
    the wipe-to-defaults is never silent (2026-06-01).
    """
    loaded = load_json(str(path), default=None)
    if loaded is not None:
        return _validated_state(loaded)

    # File is missing or corrupt — check if the file exists (corruption vs missing)
    if path.exists():
        # Capture the corrupt bytes NOW, before the recovery loop and before any
        # concurrent writer can os.replace fresh defaults onto the path (a
        # lockless load_state() reader can race a lock-held update_state writer).
        try:
            corrupt_bytes = path.read_bytes()
        except OSError:
            corrupt_bytes = b""
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
        # All backups failed — preserve evidence + surface LOUDLY before the
        # caller's next save silently overwrites the corrupt file with defaults.
        _quarantine_corrupt_state(path, corrupt_bytes)
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
