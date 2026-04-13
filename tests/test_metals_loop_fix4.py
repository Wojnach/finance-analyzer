"""Tests for Fix 4 MVP: dynamic news cadence + fast-tick microstructure.

These guard the staleness-reduction changes that landed alongside the
2026-04-13 silver intraday-dip observation: news polled 30→5 min when
silver is held; XAG orderbook snapshots accumulated every 10s during
the fast-tick instead of only once per 60s cycle.
"""
from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest


def _reload_metals_loop():
    """Re-import metals_loop so env-var-driven module constants pick up
    the test-time os.environ overrides. Returns the freshly loaded module."""
    import data.metals_loop as ml
    importlib.reload(ml)
    return ml


class TestNewsCadence:
    """Verify the news polling cadence flips between idle and active."""

    def test_default_cadence_constants(self):
        ml = importlib.import_module("data.metals_loop")
        # Idle (no active silver) — 30 min
        assert ml.NEWS_FETCH_INTERVAL == 1800
        # Active silver — 5 min default (300 s)
        assert ml.NEWS_FETCH_INTERVAL_ACTIVE_SILVER == 300

    def test_env_var_override_for_active_silver(self):
        with patch.dict(os.environ, {"NEWS_POLL_SEC_ACTIVE_SILVER": "120"}):
            ml = _reload_metals_loop()
            try:
                assert ml.NEWS_FETCH_INTERVAL_ACTIVE_SILVER == 120
            finally:
                # Reset to default so other tests aren't affected
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("NEWS_POLL_SEC_ACTIVE_SILVER", None)
                    _reload_metals_loop()

    def test_active_silver_interval_is_smaller_than_idle(self):
        ml = importlib.import_module("data.metals_loop")
        assert ml.NEWS_FETCH_INTERVAL_ACTIVE_SILVER < ml.NEWS_FETCH_INTERVAL

    def test_news_budget_envelope_documented(self):
        """At 5-min cadence, 24 h × 12 calls/h = 288 calls/day worst case.

        WARNING: NewsAPI free tier cap is 100/day. If a silver position is
        held continuously beyond ~8.3 h (100 / 12 calls per hour), the
        loop will exceed the daily quota and subsequent fetches will fail
        for the rest of the day with no automatic recovery. Mitigations:
          - Most positions exit well within 8.3 h (EOD rule fires at 21:30
            CET; intraday targets typically hit in 1-3 h).
          - Cache hits inside _fetch_metals_news() do not count toward
            the network quota.
          - If the cap becomes a problem, upgrade to a paid plan or
            switch source.

        This test asserts the worst-case envelope so future cadence
        tightening must explicitly acknowledge the budget impact.
        """
        ml = importlib.import_module("data.metals_loop")
        worst_case_per_day = (24 * 3600) // ml.NEWS_FETCH_INTERVAL_ACTIVE_SILVER
        assert worst_case_per_day == 288, (
            f"worst-case news polls/day = {worst_case_per_day}; "
            "if you change NEWS_FETCH_INTERVAL_ACTIVE_SILVER, update the "
            "NewsAPI budget reasoning in metals_loop.py and this test."
        )
        # Maximum continuous-hold duration before hitting NewsAPI free tier:
        newsapi_free_cap_per_day = 100
        max_hold_hours = newsapi_free_cap_per_day / (3600 / ml.NEWS_FETCH_INTERVAL_ACTIVE_SILVER)
        assert 8.0 <= max_hold_hours <= 9.0, (
            f"continuous hold-time-to-cap = {max_hold_hours:.1f}h; "
            "should be ~8.3h. If this drifts, the budget assumption broke."
        )


class TestFastTickOrderbookSnapshot:
    """Verify the silver fast-tick triggers an XAG orderbook snapshot."""

    def test_default_fast_tick_orderbook_enabled(self):
        ml = importlib.import_module("data.metals_loop")
        assert ml._FAST_TICK_ORDERBOOK is True

    def test_env_var_can_disable_fast_tick_orderbook(self):
        with patch.dict(os.environ, {"ORDERBOOK_FAST_TICK": "0"}):
            ml = _reload_metals_loop()
            try:
                assert ml._FAST_TICK_ORDERBOOK is False
            finally:
                os.environ.pop("ORDERBOOK_FAST_TICK", None)
                _reload_metals_loop()

    def test_single_ticker_helper_calls_underlying_collector(self):
        """_accumulate_orderbook_snapshot_for(ticker) must call
        get_orderbook_depth + accumulate_snapshot for that one ticker only."""
        ml = importlib.import_module("data.metals_loop")
        if not ml._MICROSTRUCTURE_AVAILABLE:
            pytest.skip("microstructure deps not available in this env")

        with patch.object(ml, "get_orderbook_depth") as mock_depth, \
             patch.object(ml, "accumulate_snapshot") as mock_acc:
            mock_depth.return_value = {"bids": [], "asks": []}
            ml._accumulate_orderbook_snapshot_for("XAG-USD")
            mock_depth.assert_called_once_with("XAG-USD", limit=20)
            mock_acc.assert_called_once()

    def test_single_ticker_helper_swallows_exceptions(self):
        """Failures inside the snapshot collector must NOT propagate —
        the silver fast-tick path can't tolerate exceptions because price
        threshold alerts run after this call."""
        ml = importlib.import_module("data.metals_loop")
        if not ml._MICROSTRUCTURE_AVAILABLE:
            pytest.skip("microstructure deps not available")
        with patch.object(ml, "get_orderbook_depth", side_effect=RuntimeError("FAPI down")):
            # Must not raise — asserts no exception
            ml._accumulate_orderbook_snapshot_for("XAG-USD")

    def test_helper_uses_dedicated_throttle_counter(self):
        """The fast-tick helper must increment its own counter, not the
        cycle-level _microstructure_persist_counter. Without this, the
        throttle gate would freeze at whatever modulo the cycle counter
        last left it on."""
        ml = importlib.import_module("data.metals_loop")
        if not ml._MICROSTRUCTURE_AVAILABLE:
            pytest.skip("microstructure deps not available")
        before_cycle = ml._microstructure_persist_counter
        before_for = ml._snapshot_for_call_counter
        with patch.object(ml, "get_orderbook_depth", side_effect=RuntimeError("FAPI down")):
            ml._accumulate_orderbook_snapshot_for("XAG-USD")
            ml._accumulate_orderbook_snapshot_for("XAG-USD")
            ml._accumulate_orderbook_snapshot_for("XAG-USD")
        # Cycle-level counter UNTOUCHED by the fast-tick helper:
        assert ml._microstructure_persist_counter == before_cycle
        # Fast-tick counter advanced once per failure:
        assert ml._snapshot_for_call_counter == before_for + 3

    def test_fast_tick_skips_orderbook_when_env_disabled(self):
        """When ORDERBOOK_FAST_TICK=0, the fast-tick must NOT call the
        snapshot helper even if microstructure is available. We exercise
        the gate, not the surrounding tick body, so other dependencies
        are stubbed."""
        with patch.dict(os.environ, {"ORDERBOOK_FAST_TICK": "0"}):
            ml = _reload_metals_loop()
            try:
                # Make _silver_fast_tick believe we have an active silver position
                ml._get_active_silver = MagicMock(return_value=("silver_sg", {"entry": 14.6, "units": 68, "leverage": 4.76}))
                ml._silver_fetch_xag = MagicMock(return_value=74.0)
                ml._silver_init_ref = MagicMock()
                ml._silver_underlying_ref = 74.32
                with patch.object(ml, "_accumulate_orderbook_snapshot_for") as mock_snap:
                    ml._silver_fast_tick()
                    mock_snap.assert_not_called()
            finally:
                os.environ.pop("ORDERBOOK_FAST_TICK", None)
                _reload_metals_loop()
