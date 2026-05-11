"""Verify the 2026-05-11 persistence-check dedup: SIGNAL_PERSISTENCE_CHECKS,
MACD_IMPROVING_CHECKS, REGIME_CONFIRM_CHECKS = 1 across metals/crypto/oil.

The engine-layer already filters single-cycle flips; the swing-layer set
these constants to 1 to avoid double-counting.
"""
from __future__ import annotations

import pytest

from data import crypto_swing_config as crypto_cfg
from data import oil_swing_config as oil_cfg


@pytest.mark.parametrize("cfg_name,cfg", [
    ("crypto", crypto_cfg),
    ("oil", oil_cfg),
])
def test_persistence_checks_set_to_one(cfg_name, cfg):
    assert cfg.SIGNAL_PERSISTENCE_CHECKS == 1, (
        f"{cfg_name}: SIGNAL_PERSISTENCE_CHECKS must be 1 (2026-05-11 dedup)"
    )
    assert cfg.MACD_IMPROVING_CHECKS == 1, (
        f"{cfg_name}: MACD_IMPROVING_CHECKS must be 1 (2026-05-11 dedup)"
    )
    assert cfg.REGIME_CONFIRM_CHECKS == 1, (
        f"{cfg_name}: REGIME_CONFIRM_CHECKS must be 1 (2026-05-11 dedup)"
    )


def test_metals_persistence_checks_set_to_one():
    """metals_swing_config is imported as a script-style module, not via package path."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    import metals_swing_config as metals_cfg

    assert metals_cfg.SIGNAL_PERSISTENCE_CHECKS == 1
    assert metals_cfg.MACD_IMPROVING_CHECKS == 1
    assert metals_cfg.REGIME_CONFIRM_CHECKS == 1
