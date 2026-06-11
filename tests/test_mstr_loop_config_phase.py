"""PHASE env validation at import (audit B8 fix 4).

config.PHASE is read from MSTR_LOOP_PHASE; an unknown/typo'd value used to
slip past every execution branch (and _handle_partial_sell would mutate
state with no order). The module now validates at import and fail-loud.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _reimport_config(monkeypatch, phase_value):
    monkeypatch.setenv("MSTR_LOOP_PHASE", phase_value)
    sys.modules.pop("portfolio.mstr_loop.config", None)
    return importlib.import_module("portfolio.mstr_loop.config")


@pytest.fixture(autouse=True)
def _restore_config():
    yield
    # Reload a clean default-phase config so other tests see canonical state.
    sys.modules.pop("portfolio.mstr_loop.config", None)
    importlib.import_module("portfolio.mstr_loop.config")


@pytest.mark.parametrize("phase", ["shadow", "paper", "live"])
def test_valid_phases_import_cleanly(monkeypatch, phase):
    cfg = _reimport_config(monkeypatch, phase)
    assert cfg.PHASE == phase


@pytest.mark.parametrize("bad", ["Paper", "prod", "PROD", "shadow ", "", "liv"])
def test_unknown_phase_raises_at_import(monkeypatch, bad):
    # Note: "shadow " strips to "shadow" (valid); the env raw has trailing
    # space but config strips before validating, so only truly-unknown
    # values raise. Filter to values that survive .strip() as unknown.
    monkeypatch.setenv("MSTR_LOOP_PHASE", bad)
    sys.modules.pop("portfolio.mstr_loop.config", None)
    stripped = bad.strip()
    if stripped in ("shadow", "paper", "live"):
        cfg = importlib.import_module("portfolio.mstr_loop.config")
        assert cfg.PHASE == stripped
        return
    if stripped == "":
        # Empty -> falls back to "shadow" via the `or "shadow"` default.
        cfg = importlib.import_module("portfolio.mstr_loop.config")
        assert cfg.PHASE == "shadow"
        return
    with pytest.raises(ValueError, match="not a valid phase"):
        importlib.import_module("portfolio.mstr_loop.config")
