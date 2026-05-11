"""Tests for MSTR loop low-cash sizing mode (2026-05-11).

When cash < LOW_CASH_THRESHOLD_SEK, MIN_TRADE_SEK becomes the *target*
position size rather than a floor applied on top of POSITION_SIZE_PCT.

Mirrors the metals/crypto/oil swing-trader sizing pattern so small-ISK
runs (e.g. after cash drawdown) don't underbuy below the Avanza courtage
minimum.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.mstr_loop import config, execution
from portfolio.mstr_loop.state import BotState


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Redirect config paths so tests never touch live files."""
    monkeypatch.setattr(config, "STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setattr(config, "TRADES_LOG", str(tmp_path / "trades.jsonl"))
    monkeypatch.setattr(config, "SHADOW_LOG", str(tmp_path / "shadow.jsonl"))
    monkeypatch.setattr(config, "POLL_LOG", str(tmp_path / "poll.jsonl"))
    yield


@pytest.fixture(autouse=True)
def _standard_sizing(monkeypatch):
    """Pin sizing knobs so tests are independent of config-file drift."""
    monkeypatch.setattr(config, "PHASE", "paper")
    monkeypatch.setattr(config, "MIN_TRADE_SEK", 1000)
    monkeypatch.setattr(config, "LOW_CASH_THRESHOLD_SEK", 10_000)
    monkeypatch.setattr(config, "POSITION_SIZE_PCT", 30)
    yield


# ---------------------------------------------------------------------------
# Low-cash mode: cash < LOW_CASH_THRESHOLD_SEK → MIN_TRADE_SEK is the target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cash, expected",
    [
        # Very low — even MIN_TRADE_SEK can't fit under the 95% cap → skip
        (500, 0.0),   # 500 * 0.95 = 475 < 1000 floor → 0
        (900, 0.0),   # 900 * 0.95 = 855 < 1000 floor → 0
        (1000, 0.0),  # 1000 * 0.95 = 950 < 1000 floor → 0
        # Just enough that 95% × cash ≥ MIN_TRADE_SEK → target the floor
        (1100, 1000.0),  # min(1000, 1045) = 1000
        (1500, 1000.0),  # min(1000, 1425) = 1000
        (2822, 1000.0),  # known regression case from metals: 2822 × 30% = 847 → 1000
        (5000, 1000.0),  # would be 1500 in normal mode, low-cash forces 1000
        (9999, 1000.0),  # just under threshold — still low-cash
    ],
)
def test_low_cash_mode_targets_min_trade_sek(cash, expected):
    """When cash < LOW_CASH_THRESHOLD_SEK, position size is MIN_TRADE_SEK
    (capped at 95% of cash, returning 0 if floor doesn't clear cap)."""
    result = execution._notional_for_entry(BotState(cash_sek=cash))
    assert result == pytest.approx(expected), (
        f"cash={cash} expected {expected}, got {result}"
    )


# ---------------------------------------------------------------------------
# High-cash mode: cash >= LOW_CASH_THRESHOLD_SEK → POSITION_SIZE_PCT applies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cash, expected",
    [
        # Just over threshold: 10K × 30% = 3000, above 1000 floor
        (10_000, 3000.0),
        (20_000, 6000.0),
        (50_000, 15_000.0),
        (100_000, 30_000.0),
    ],
)
def test_high_cash_mode_uses_position_size_pct(cash, expected):
    """When cash ≥ LOW_CASH_THRESHOLD_SEK, normal POSITION_SIZE_PCT sizing."""
    result = execution._notional_for_entry(BotState(cash_sek=cash))
    assert result == pytest.approx(expected)


def test_threshold_boundary_is_inclusive_of_high_branch():
    """At exactly the threshold, we take the high-cash branch
    (cash < THRESHOLD is strict)."""
    # cash = 10_000, threshold = 10_000 → strict `<` is False → high-cash
    result = execution._notional_for_entry(BotState(cash_sek=10_000))
    # 30% × 10K = 3000, above 1000 floor → 3000
    assert result == pytest.approx(3000.0)


def test_shadow_phase_unaffected_by_low_cash(monkeypatch):
    """Shadow phase uses SHADOW_NOTIONAL_SEK regardless of cash."""
    monkeypatch.setattr(config, "PHASE", "shadow")
    monkeypatch.setattr(config, "SHADOW_NOTIONAL_SEK", 30_000)
    # Even with low cash, shadow returns the hypothetical notional
    assert execution._notional_for_entry(BotState(cash_sek=100)) == 30_000
    assert execution._notional_for_entry(BotState(cash_sek=1000)) == 30_000
    assert execution._notional_for_entry(BotState(cash_sek=50_000)) == 30_000


def test_live_phase_uses_same_low_cash_logic_as_paper(monkeypatch):
    """live and paper share the same low-cash sizing math (only execution
    side differs). Regression guard against accidental divergence."""
    monkeypatch.setattr(config, "PHASE", "live")
    # cash=5000 < 10K threshold → MIN_TRADE_SEK target
    assert execution._notional_for_entry(BotState(cash_sek=5000)) == 1000
    # cash=50K ≥ 10K → POSITION_SIZE_PCT
    assert execution._notional_for_entry(BotState(cash_sek=50_000)) == 15_000


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------


def test_low_cash_threshold_is_configurable(monkeypatch):
    """Operator can tune the threshold via config edit (not env)."""
    monkeypatch.setattr(config, "LOW_CASH_THRESHOLD_SEK", 5000)
    # cash=8000 was low-cash before (< 10K), now high-cash (> 5K)
    # high-cash: 30% × 8000 = 2400, above floor → 2400
    assert execution._notional_for_entry(BotState(cash_sek=8000)) == pytest.approx(2400.0)


def test_low_cash_threshold_default_value():
    """The default threshold matches the spec'd 10_000 SEK."""
    assert config.LOW_CASH_THRESHOLD_SEK == 10_000
