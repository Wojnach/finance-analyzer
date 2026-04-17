"""Unit tests for the entry-side fast-tick (upside-momentum detector).

Covers the 2026-04-17 feature that mirrors ``_silver_fast_tick`` in the
opposite direction, not gated on having an active position. Tests exercise
``_entry_fast_tick`` directly with a seeded deque and fake fetch function
so no network calls are made.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque

import pytest

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_loop as ml


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Redirect the momentum state file to tmp_path and reset the dedup map.

    Module-level state carries between tests (xdist-unsafe if left alone);
    each test must start with an empty deque and an empty dedup map so the
    dedup / threshold branches are deterministic.
    """
    state_file = tmp_path / "metals_momentum_state.json"
    monkeypatch.setattr(ml, "ENTRY_MOMENTUM_STATE_FILE", str(state_file))
    monkeypatch.setattr(ml, "_entry_last_trigger_ts", {})
    # Disable Telegram to avoid any outbound during tests.
    monkeypatch.setattr(ml, "ENTRY_MOMENTUM_TELEGRAM", False)
    yield


def _seed_rising_prices(n: int, start: float, per_tick_pct: float) -> list[float]:
    """Return n rising prices, each (per_tick_pct) percent above the prior."""
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * (1 + per_tick_pct / 100))
    return out


def _make_deque(prices: list[float], maxlen: int = 18) -> deque:
    d = deque(maxlen=maxlen)
    for p in prices:
        d.append(p)
    return d


def test_entry_fast_tick_writes_candidate_on_positive_velocity(monkeypatch, tmp_path):
    """18-tick window with +1.0% total rise + rvol=2.0 should write a candidate."""
    # Seed 17 prices rising cumulatively to just below +1.0% so the deque
    # is one slot short of full; the _entry_fast_tick call appends the 18th.
    prices = _seed_rising_prices(17, start=78.00, per_tick_pct=0.05)  # rises ~0.85%
    d = _make_deque(prices, maxlen=18)
    # The 18th tick lifts velocity to ~+1.0% total.
    final_price = prices[-1] * 1.002  # another +0.2% → ~+1.0% over full window

    monkeypatch.setattr(ml, "_fetch_rvol", lambda ticker: 2.0)

    ml._entry_fast_tick(
        ticker="XAG-USD",
        fetch_fn=lambda: final_price,
        prices_deque=d,
        threshold_pct=0.8,
        min_rvol=1.5,
        dedup_sec=300,
        ttl_sec=300,
    )

    state = json.load(open(ml.ENTRY_MOMENTUM_STATE_FILE, encoding="utf-8"))
    assert "XAG-USD" in state
    cand = state["XAG-USD"]
    assert cand["direction"] == "LONG"
    assert cand["velocity_pct"] >= 0.8
    assert cand["rvol"] == 2.0
    assert cand["price_at_trigger"] == round(final_price, 4)
    assert cand["consumed_at"] is None
    assert cand["ttl_sec"] == 300


def test_entry_fast_tick_skips_when_velocity_below_threshold(monkeypatch):
    """+0.3% total rise is below the 0.8% threshold → no candidate written."""
    prices = _seed_rising_prices(17, start=78.00, per_tick_pct=0.01)  # ~+0.16%
    d = _make_deque(prices, maxlen=18)
    final_price = prices[-1] * 1.001  # another +0.1% → total ~+0.26%

    monkeypatch.setattr(ml, "_fetch_rvol", lambda ticker: 2.5)  # high rvol, irrelevant

    ml._entry_fast_tick(
        ticker="XAG-USD",
        fetch_fn=lambda: final_price,
        prices_deque=d,
        threshold_pct=0.8,
        min_rvol=1.5,
        dedup_sec=300,
    )

    assert not os.path.exists(ml.ENTRY_MOMENTUM_STATE_FILE)


def test_entry_fast_tick_dedup_window(monkeypatch):
    """Two consecutive triggers within dedup_sec → only the first writes."""
    prices = _seed_rising_prices(17, start=78.00, per_tick_pct=0.06)
    d = _make_deque(prices, maxlen=18)
    final_price = prices[-1] * 1.003  # total rise >= 1%

    monkeypatch.setattr(ml, "_fetch_rvol", lambda ticker: 2.0)

    ml._entry_fast_tick(
        ticker="XAG-USD",
        fetch_fn=lambda: final_price,
        prices_deque=d,
        threshold_pct=0.8,
        min_rvol=1.5,
        dedup_sec=300,
    )
    assert os.path.exists(ml.ENTRY_MOMENTUM_STATE_FILE)
    mtime_first = os.path.getmtime(ml.ENTRY_MOMENTUM_STATE_FILE)

    # Second call immediately — still in dedup window. Advance deque by one
    # more tick to keep velocity valid, then call again.
    time.sleep(0.05)
    d.append(final_price * 1.001)
    ml._entry_fast_tick(
        ticker="XAG-USD",
        fetch_fn=lambda: final_price * 1.002,
        prices_deque=d,
        threshold_pct=0.8,
        min_rvol=1.5,
        dedup_sec=300,
    )
    mtime_second = os.path.getmtime(ml.ENTRY_MOMENTUM_STATE_FILE)

    assert mtime_first == mtime_second, (
        "State file must not have been rewritten during dedup window"
    )


def test_entry_fast_tick_requires_rvol(monkeypatch):
    """Velocity clears the bar but rvol=0.8 → candidate suppressed."""
    prices = _seed_rising_prices(17, start=78.00, per_tick_pct=0.06)
    d = _make_deque(prices, maxlen=18)
    final_price = prices[-1] * 1.002

    monkeypatch.setattr(ml, "_fetch_rvol", lambda ticker: 0.8)  # below min

    ml._entry_fast_tick(
        ticker="XAG-USD",
        fetch_fn=lambda: final_price,
        prices_deque=d,
        threshold_pct=0.8,
        min_rvol=1.5,
        dedup_sec=300,
    )

    assert not os.path.exists(ml.ENTRY_MOMENTUM_STATE_FILE), (
        "Low rvol should suppress candidate even when velocity clears threshold"
    )


def test_entry_fast_tick_skips_until_deque_full(monkeypatch):
    """With fewer than maxlen ticks, velocity cannot be computed → skip."""
    d = deque(maxlen=18)
    d.extend([78.00, 78.05, 78.10])  # only 3 of 18

    monkeypatch.setattr(ml, "_fetch_rvol", lambda ticker: 10.0)  # would pass

    ml._entry_fast_tick(
        ticker="XAG-USD",
        fetch_fn=lambda: 79.50,  # +1.8% from 78.00 — would trigger if ready
        prices_deque=d,
        threshold_pct=0.8,
        min_rvol=1.5,
        dedup_sec=300,
    )

    assert not os.path.exists(ml.ENTRY_MOMENTUM_STATE_FILE), (
        "Must not write candidate until deque is fully populated"
    )


def test_entry_fast_tick_preserves_other_ticker_candidates(monkeypatch):
    """Writing an XAG candidate must not clobber an existing XAU candidate."""
    # Pre-seed an XAU candidate in state.
    ml._write_momentum_candidate(
        ticker="XAU-USD",
        direction="LONG",
        velocity_pct=0.55,
        price=4780.0,
        rvol=2.1,
        ttl_sec=300,
    )
    assert os.path.exists(ml.ENTRY_MOMENTUM_STATE_FILE)

    # Now fire an XAG trigger.
    prices = _seed_rising_prices(17, start=78.00, per_tick_pct=0.06)
    d = _make_deque(prices, maxlen=18)
    final_price = prices[-1] * 1.003

    monkeypatch.setattr(ml, "_fetch_rvol", lambda ticker: 2.0)

    ml._entry_fast_tick(
        ticker="XAG-USD",
        fetch_fn=lambda: final_price,
        prices_deque=d,
        threshold_pct=0.8,
        min_rvol=1.5,
        dedup_sec=300,
    )

    state = json.load(open(ml.ENTRY_MOMENTUM_STATE_FILE, encoding="utf-8"))
    assert "XAG-USD" in state
    assert "XAU-USD" in state, "XAU candidate must be preserved when writing XAG"
    assert state["XAU-USD"]["velocity_pct"] == 0.55


def test_entry_fast_tick_handles_fetch_failure(monkeypatch):
    """If fetch_fn returns None, tick must return cleanly without writing."""
    prices = _seed_rising_prices(17, start=78.00, per_tick_pct=0.06)
    d = _make_deque(prices, maxlen=18)

    ml._entry_fast_tick(
        ticker="XAG-USD",
        fetch_fn=lambda: None,  # simulate network failure
        prices_deque=d,
        threshold_pct=0.8,
        min_rvol=1.5,
        dedup_sec=300,
    )

    assert not os.path.exists(ml.ENTRY_MOMENTUM_STATE_FILE)


def test_sleep_for_cycle_runs_entry_tick_without_silver_position(monkeypatch):
    """The key architectural gap: entry ticks must run regardless of position.

    Before 2026-04-17 ``_sleep_for_cycle`` short-circuited when
    ``_has_active_silver()`` returned False, leaving the fast-tick machinery
    dark and unable to detect breakout entries. This test confirms the new
    gate lets the entry tick fire.
    """
    called = {"silver_entry": 0, "gold_entry": 0, "silver_exit": 0}

    monkeypatch.setattr(ml, "_has_active_silver", lambda: False)
    monkeypatch.setattr(ml, "SILVER_ENTRY_FAST_TICK_ENABLED", True)
    monkeypatch.setattr(ml, "GOLD_ENTRY_FAST_TICK_ENABLED", True)
    monkeypatch.setattr(
        ml, "_silver_entry_fast_tick",
        lambda: called.__setitem__("silver_entry", called["silver_entry"] + 1),
    )
    monkeypatch.setattr(
        ml, "_gold_entry_fast_tick",
        lambda: called.__setitem__("gold_entry", called["gold_entry"] + 1),
    )
    monkeypatch.setattr(
        ml, "_silver_fast_tick",
        lambda: called.__setitem__("silver_exit", called["silver_exit"] + 1),
    )
    # Shrink the tick interval so the test finishes quickly.
    monkeypatch.setattr(ml, "SILVER_FAST_TICK_INTERVAL", 0.05)

    cycle_started = time.monotonic()
    ml._sleep_for_cycle(cycle_started, interval_s=0.2, label="test")

    assert called["silver_entry"] >= 1, "entry tick must fire even with no position"
    assert called["gold_entry"] >= 1, "gold entry tick must fire even with no position"
    assert called["silver_exit"] == 0, "exit tick must NOT fire without active silver"
