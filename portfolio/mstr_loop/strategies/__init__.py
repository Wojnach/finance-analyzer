"""Strategy registry for MSTR Loop.

Each strategy is a module under this package exposing a module-level
instance named `STRATEGY`. The loop imports all enabled strategies at
startup via `load_enabled_strategies()`.

Adding a new strategy: drop a file here, export a STRATEGY instance, add
its key to `config.STRATEGY_TOGGLES` with `True`. No other plumbing
required.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from portfolio.mstr_loop import config

if TYPE_CHECKING:
    from portfolio.mstr_loop.strategies.base import Strategy

logger = logging.getLogger(__name__)


def load_enabled_strategies() -> list["Strategy"]:
    """Import and return instances of every strategy toggled on in config."""
    strategies: list[Strategy] = []
    for key, enabled in config.STRATEGY_TOGGLES.items():
        if not enabled:
            continue
        try:
            mod = importlib.import_module(f"portfolio.mstr_loop.strategies.{key}")
            strat = getattr(mod, "STRATEGY", None)
            if strat is None:
                logger.warning("strategies: %s has no STRATEGY export — skipping", key)
                continue
            strategies.append(strat)
        except ImportError:
            logger.warning("strategies: %s module missing — skipping", key, exc_info=True)
        except Exception:
            logger.exception("strategies: failed to load %s — skipping", key)
    return strategies
