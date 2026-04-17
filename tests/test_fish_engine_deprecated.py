"""Regression + guard tests for the fish engine deprecation (2026-04-17).

The fish engine was disabled 2026-04-15 after losing 12,257 SEK in one
session. Tests lock in:

- FISH_ENGINE_ENABLED stays False at import.
- _assert_fish_engine_allowed raises if the flag is flipped to True.
- _run_fish_engine_tick calls the assertion before any work.

Together these force a future operator to make TWO deliberate changes to
re-activate the engine: flip the flag AND remove the assertion. A single-
line change cannot silently reactivate a known-losing strategy.
"""

from __future__ import annotations

import os
import sys

import pytest

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_loop as ml


def test_fish_engine_flag_stays_false():
    """Regression guard: FISH_ENGINE_ENABLED must stay False at module import.

    If a future operator flips it, this test catches it in CI even if
    they forget to delete the assertion guard.
    """
    assert ml.FISH_ENGINE_ENABLED is False, (
        "Fish engine is deprecated (2026-04-17). To revive, see the "
        "docstring at the declaration site. Do NOT simply flip this flag."
    )


def test_assert_fish_engine_allowed_does_not_raise_when_disabled():
    """With FISH_ENGINE_ENABLED=False, the guard is a no-op."""
    # Must not raise; returns None.
    result = ml._assert_fish_engine_allowed()
    assert result is None


def test_assert_fish_engine_allowed_raises_when_flag_flipped(monkeypatch):
    """Flipping FISH_ENGINE_ENABLED=True makes the guard raise loudly."""
    monkeypatch.setattr(ml, "FISH_ENGINE_ENABLED", True)
    with pytest.raises(RuntimeError) as exc_info:
        ml._assert_fish_engine_allowed()
    msg = str(exc_info.value)
    # Must name the 2026-04-17 deprecation and mention the 12,257 SEK loss
    # so a confused operator reading the traceback gets context directly.
    assert "2026-04-17" in msg
    assert "12,257" in msg or "deprecated" in msg.lower()


def test_run_fish_engine_tick_raises_when_flag_flipped(monkeypatch):
    """_run_fish_engine_tick calls the guard before any work.

    This is the critical gate: even if the outer `if FISH_ENGINE_ENABLED:`
    check in the main loop is tampered with, the tick function itself
    refuses to run without the assertion being removed.
    """
    monkeypatch.setattr(ml, "FISH_ENGINE_ENABLED", True)
    with pytest.raises(RuntimeError):
        ml._run_fish_engine_tick()


def test_assert_is_a_callable_function():
    """The guard must be a function (callable) not an expression — ensures
    it's testable and that call sites can be grepped/audited.
    """
    assert callable(ml._assert_fish_engine_allowed)
