"""Master kill switch for ALL live order placement.

The operator's standing instruction (2026-08-05): the system monitors, computes
and predicts; every buy and sell is made by hand. This gate enforces that in
code rather than relying on each subsystem's own dry_run flag, because those
flags are per-module, default differently, and are easy to miss — golddigger
had `trade_enabled: false` while elongir had no trading flag at all, and
`portfolio/main.py` reached a live order path through
`avanza_orders.check_pending_orders -> _execute_confirmed_order`.

Placed at the lowest level (the functions that actually POST to Avanza) so that
every caller is covered no matter which path it takes.

**Fails CLOSED.** If the flag file cannot be read, or the config cannot be
parsed, or anything at all goes wrong in this module, trading is treated as
disabled. A kill switch that fails open is not a kill switch.

Disable (default state)::

    touch data/trading.disabled

Re-enable — deliberately requires BOTH removing the flag AND an explicit config
key, so no single accidental `rm` can arm live trading::

    rm data/trading.disabled
    # and set trading.live_enabled = true in config.json
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("portfolio.trading_gate")

FLAG_FILE = "data/trading.disabled"
CONFIG_KEY = "live_enabled"
CONFIG_SECTION = "trading"


class TradingDisabledError(RuntimeError):
    """Raised instead of placing an order while the gate is closed."""


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def flag_present(path=None):
    try:
        return os.path.exists(path or os.path.join(_repo_root(), FLAG_FILE))
    except Exception:
        # Cannot determine -> assume the operator meant to disable.
        return True


def config_allows():
    try:
        from portfolio.file_utils import load_json

        cfg = load_json(os.path.join(_repo_root(), "config.json")) or {}
        return bool((cfg.get(CONFIG_SECTION) or {}).get(CONFIG_KEY, False))
    except Exception:
        return False


def trading_enabled():
    """True only when the flag is absent AND config explicitly opts in."""
    try:
        if flag_present():
            return False
        return config_allows()
    except Exception:
        logger.error("trading_gate: check failed — treating as DISABLED")
        return False


def reason():
    if flag_present():
        return f"kill-switch file present ({FLAG_FILE})"
    if not config_allows():
        return f"config {CONFIG_SECTION}.{CONFIG_KEY} is not true"
    return "enabled"


def require_trading_enabled(op="order"):
    """Guard for any function that places, modifies or cancels a live order.

    Raises rather than returning a falsy value: a caller that ignores a return
    code would go on to place the order, and silence is the failure mode this
    exists to prevent.
    """
    if trading_enabled():
        return True
    msg = (
        f"live trading is DISABLED — refusing {op}. Reason: {reason()}. "
        f"The operator places all orders by hand; to re-arm, remove "
        f"{FLAG_FILE} AND set {CONFIG_SECTION}.{CONFIG_KEY}=true in config.json."
    )
    logger.error("trading_gate: %s", msg)
    raise TradingDisabledError(msg)
