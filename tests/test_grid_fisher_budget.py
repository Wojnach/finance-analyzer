"""Buying-power gate tests for GridFisher.

Covers the 2026-05-13 fix that wires ``_effective_global_cap`` into the
tick(). Previously the cap was the static config value
(GRID_GLOBAL_MAX_SEK=6500) regardless of actual Avanza buying power;
grid_fisher would happily attempt OLJAB / silver / gold placements
even when the account only had a few thousand SEK available.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import pytest

from portfolio import grid_fisher_config as gfc
from portfolio.grid_fisher import GridFisher


class _StubSession:
    """Minimal session double exposing the methods the GridFisher uses.

    The tests poke ``buying_power_return`` / ``buying_power_raises`` to
    simulate the live-cash gate path. Order placement is captured for
    assertions.
    """

    def __init__(self, buying_power_return: Optional[dict] = None,
                 buying_power_raises: bool = False) -> None:
        self.buying_power_return = buying_power_return
        self.buying_power_raises = buying_power_raises
        self.buying_power_calls = 0
        self.place_buy_calls: list[tuple[str, float, int]] = []
        self.open_orders: list[dict] = []
        self.positions: list[dict] = []

    def get_buying_power(self, account_id: str):
        self.buying_power_calls += 1
        if self.buying_power_raises:
            raise RuntimeError("simulated avanza failure")
        return self.buying_power_return

    def get_open_orders(self):
        return list(self.open_orders)

    def get_positions(self):
        return list(self.positions)

    def get_quote(self, ob_id: str):
        # ``timeOfLast`` keeps the new Gate A (quote-staleness) from
        # short-circuiting these tests with a closed-orderbook decision.
        return {"buy": 100.0, "sell": 100.5,
                "timeOfLast": int(time.time() * 1000)}

    def place_buy_order(self, ob_id: str, price: float, volume: int):
        self.place_buy_calls.append((ob_id, price, volume))
        return {"orderRequestStatus": "SUCCESS",
                "orderId": f"sim-{len(self.place_buy_calls)}"}

    def place_sell_order(self, ob_id: str, price: float, volume: int):
        return {"orderRequestStatus": "SUCCESS", "orderId": "sim-sell"}

    def place_stop_loss(self, *args, **kwargs):
        return {"orderRequestStatus": "SUCCESS", "orderId": "sim-stop"}

    def cancel_order(self, order_id: str):
        return {"orderRequestStatus": "SUCCESS"}


@pytest.fixture
def grid_catalog() -> dict[str, dict[str, Any]]:
    """Single XAG/LONG instrument so the fisher seeds exactly one slot.

    Keeps assertions clear — the test isn't about multi-instrument
    interactions, only about the cap arithmetic.
    """
    return {
        "1650161": {
            "ticker": "XAG-USD",
            "direction": "LONG",
            "name": "BULL_SILVER_X5_AVA_4",
            "barrier": 18.0,
            "leverage": 5.0,
        },
    }


@pytest.fixture
def single_instrument_active(monkeypatch, grid_catalog):
    """Restrict GRID_ACTIVE_INSTRUMENTS to a single XAG/LONG slot so seeding
    is deterministic regardless of config drift."""
    monkeypatch.setattr(
        gfc, "GRID_ACTIVE_INSTRUMENTS",
        {"XAG-USD": {"LONG": "1650161"}},
    )


def _make_fisher(session, tmp_path, account_id: Optional[str] = "1625505"):
    return GridFisher(
        session=session,
        catalog={
            "1650161": {
                "ticker": "XAG-USD",
                "direction": "LONG",
                "barrier": 18.0,
                "leverage": 5.0,
            },
        },
        state_path=tmp_path / "grid_state.json",
        decisions_path=tmp_path / "grid_decisions.jsonl",
        account_id=account_id,
        signal_fn=lambda *_a, **_k: {"direction": "LONG", "confidence": 0.9},
        atr_fn=lambda *_a, **_k: 2.0,
        adx_fn=lambda *_a, **_k: 15.0,
    )


# ---- _effective_global_cap arithmetic ------------------------------------


class TestEffectiveCap:
    def test_no_account_id_returns_config_cap(self, tmp_path):
        """When the caller didn't wire account_id, behaviour falls back to
        the previous static cap. Matches the back-compat contract."""
        sess = _StubSession()
        f = _make_fisher(sess, tmp_path, account_id=None)
        cap, dbg = f._effective_global_cap()
        assert cap == float(gfc.GRID_GLOBAL_MAX_SEK)
        assert dbg["source"] == "config_only"
        assert sess.buying_power_calls == 0

    def test_live_bp_above_config_cap(self, tmp_path):
        """Plenty of cash → clamped to config max, not to bp - buffer."""
        sess = _StubSession(buying_power_return={"buying_power": 50_000.0})
        f = _make_fisher(sess, tmp_path)
        cap, dbg = f._effective_global_cap()
        assert cap == float(gfc.GRID_GLOBAL_MAX_SEK)
        assert dbg["source"] == "live_buying_power"
        assert dbg["buying_power"] == 50_000

    def test_live_bp_below_config_cap_clamps_down(self, tmp_path):
        """The 2026-05-11 incident scenario: account has ~3097 SEK,
        buffer is 500, so the effective cap should be ~2597 SEK — far
        below the 6500 config cap."""
        sess = _StubSession(buying_power_return={"buying_power": 3097.0})
        f = _make_fisher(sess, tmp_path)
        cap, dbg = f._effective_global_cap()
        expected = max(0.0, 3097.0 - gfc.GRID_CASH_SAFETY_BUFFER_SEK)
        assert cap == expected
        assert cap < gfc.GRID_GLOBAL_MAX_SEK
        assert dbg["clamped"] == round(expected, 0)

    def test_zero_buying_power_returns_zero(self, tmp_path):
        sess = _StubSession(buying_power_return={"buying_power": 0.0})
        f = _make_fisher(sess, tmp_path)
        cap, _dbg = f._effective_global_cap()
        assert cap == 0.0

    def test_buying_power_below_buffer_returns_zero(self, tmp_path):
        """Cash exists but doesn't cover the safety buffer — fail-closed."""
        sess = _StubSession(
            buying_power_return={"buying_power": gfc.GRID_CASH_SAFETY_BUFFER_SEK - 1},
        )
        f = _make_fisher(sess, tmp_path)
        cap, _dbg = f._effective_global_cap()
        assert cap == 0.0


# ---- Fail-closed on fetch failure ----------------------------------------


class TestFetchFailure:
    def test_no_cache_and_fetch_returns_none_fails_closed(self, tmp_path):
        """First-ever tick, fetch fails: effective cap is 0 so the gate
        rejects every instrument. Strictly more conservative than the
        old behaviour (which used the static cap regardless)."""
        sess = _StubSession(buying_power_return=None)
        f = _make_fisher(sess, tmp_path)
        cap, dbg = f._effective_global_cap()
        assert cap == 0.0
        assert dbg["source"] == "fail_closed"

    def test_no_cache_and_session_raises_fails_closed(self, tmp_path):
        sess = _StubSession(buying_power_raises=True)
        f = _make_fisher(sess, tmp_path)
        cap, dbg = f._effective_global_cap()
        assert cap == 0.0
        assert dbg["source"] == "fail_closed"

    def test_stale_cache_within_grace_is_reused(self, tmp_path, monkeypatch):
        """If a recent reading exists and the next fetch fails, the cap
        falls back to the cached value rather than failing closed."""
        sess = _StubSession(buying_power_return={"buying_power": 5000.0})
        f = _make_fisher(sess, tmp_path)
        # Prime the cache.
        cap1, _ = f._effective_global_cap()
        assert cap1 > 0
        # Next fetch fails — but cache is fresh.
        sess.buying_power_return = None
        # Force the cache to be older than the fresh window but younger
        # than the stale-grace window so the failed fetch triggers reuse.
        cached_ts, cached_val = f._buying_power_cache
        f._buying_power_cache = (
            cached_ts - gfc.GRID_BUYING_POWER_CACHE_SECS - 1,
            cached_val,
        )
        cap2, dbg = f._effective_global_cap()
        assert cap2 == cap1  # reused
        assert dbg["source"] == "live_buying_power"

    def test_expired_cache_after_fetch_failure_fails_closed(self, tmp_path):
        sess = _StubSession(buying_power_return={"buying_power": 5000.0})
        f = _make_fisher(sess, tmp_path)
        f._effective_global_cap()  # prime
        sess.buying_power_return = None
        # Age the cache past the stale-grace window.
        cached_ts, cached_val = f._buying_power_cache
        f._buying_power_cache = (
            cached_ts - gfc.GRID_BUYING_POWER_STALE_GRACE_SECS - 1,
            cached_val,
        )
        cap, dbg = f._effective_global_cap()
        assert cap == 0.0
        assert dbg["source"] == "fail_closed"


# ---- Cache behaviour -----------------------------------------------------


class TestBuyingPowerCache:
    def test_consecutive_calls_within_window_hit_cache(self, tmp_path):
        sess = _StubSession(buying_power_return={"buying_power": 4000.0})
        f = _make_fisher(sess, tmp_path)
        f._effective_global_cap()
        f._effective_global_cap()
        f._effective_global_cap()
        assert sess.buying_power_calls == 1

    def test_expired_cache_refetches(self, tmp_path):
        sess = _StubSession(buying_power_return={"buying_power": 4000.0})
        f = _make_fisher(sess, tmp_path)
        f._effective_global_cap()
        cached_ts, cached_val = f._buying_power_cache
        f._buying_power_cache = (
            cached_ts - gfc.GRID_BUYING_POWER_CACHE_SECS - 1,
            cached_val,
        )
        f._effective_global_cap()
        assert sess.buying_power_calls == 2


# ---- End-to-end: tick() honours the live cap -----------------------------


class TestTickHonoursLiveCap:
    def test_tick_with_insufficient_cash_places_nothing(
        self, tmp_path, single_instrument_active,
    ):
        """The bug we shipped to fix: an account with ~3 097 SEK buying
        power should NOT result in 1 200-SEK ladder placements when
        even one tier would exceed (buying_power - buffer).

        With buying_power=3097 and buffer=500, effective cap is 2597.
        A single 1 200-SEK leg (GRID_LEG_SEK) is fine on the first
        placement, but the second tier of the SAME instrument should
        be blocked by the per-leg gate inside place_buy_ladder
        because the projected aggregate would land just under cap;
        the third would breach. This test asserts the GATE engages,
        not the exact number of legs (build_buy_ladder configuration
        can change). The key invariant: the tick respects live cash.
        """
        sess = _StubSession(buying_power_return={"buying_power": 100.0})
        f = _make_fisher(sess, tmp_path)
        report = f.tick(
            signal_data={"XAG-USD": {"direction": "LONG", "confidence": 0.9}},
            prices={"1650161": {"bid": 100.0, "ask": 100.5}},
        )
        assert report.get("placements", 0) == 0
        # And no orders were sent to Avanza.
        assert sess.place_buy_calls == []

    def test_tick_with_ample_cash_places_orders(
        self, tmp_path, single_instrument_active,
    ):
        sess = _StubSession(buying_power_return={"buying_power": 50_000.0})
        f = _make_fisher(sess, tmp_path)
        report = f.tick(
            signal_data={"XAG-USD": {"direction": "LONG", "confidence": 0.9}},
            prices={"1650161": {"bid": 100.0, "ask": 100.5}},
        )
        # At least one tier should land when there's plenty of cash.
        assert report.get("placements", 0) >= 1
        assert len(sess.place_buy_calls) >= 1

    def test_tick_fail_closed_when_bp_unknown(
        self, tmp_path, single_instrument_active,
    ):
        """No live cash visibility, no cached fallback → no orders."""
        sess = _StubSession(buying_power_return=None)
        f = _make_fisher(sess, tmp_path)
        report = f.tick(
            signal_data={"XAG-USD": {"direction": "LONG", "confidence": 0.9}},
            prices={"1650161": {"bid": 100.0, "ask": 100.5}},
        )
        assert report.get("placements", 0) == 0
        assert sess.place_buy_calls == []
