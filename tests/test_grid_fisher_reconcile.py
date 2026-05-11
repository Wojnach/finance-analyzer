"""Order-lifecycle tests for portfolio.grid_fisher.GridFisher.

Uses a FakeSession that records calls and returns synthetic order IDs
to avoid hitting Avanza. Covers: reconciliation (fill vs cancel
detection, partial fills, inventory drift), rotation (sell + stop after
buy fill), direction flip cancel-then-arm, rate limiting, per-instrument
cap, session loss limit, cooldown, and probe-only mode.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from portfolio import grid_fisher as gf
from portfolio.grid_fisher import (
    GridFisher,
    InstrumentState,
    ORDER_ARMED,
    ORDER_CANCELLED,
    ORDER_FILLED,
    TierOrder,
    reconcile_against_live,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeSession:
    """In-memory stand-in for portfolio.avanza_session."""

    def __init__(self) -> None:
        self._id = itertools.count(start=1)
        self.placed_buys: list[dict[str, Any]] = []
        self.placed_sells: list[dict[str, Any]] = []
        self.placed_stops: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        # Simulated live state — tests can mutate to model fills.
        self.open_orders: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []
        # Toggleable failure modes
        self.fail_on_buy = False
        self.reject_buy = False

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._id)}"

    def place_buy_order(self, orderbook_id: str, price: float, volume: int):
        if self.fail_on_buy:
            raise RuntimeError("simulated network failure")
        if self.reject_buy:
            return {"orderRequestStatus": "ERROR", "message": "rejected"}
        order_id = self._next_id("BUY")
        self.placed_buys.append({"ob_id": orderbook_id, "price": price,
                                 "qty": volume, "order_id": order_id})
        self.open_orders.append({"orderId": order_id, "orderbookId": orderbook_id,
                                 "side": "BUY", "price": price, "volume": volume})
        return {"orderRequestStatus": "SUCCESS", "orderId": order_id}

    def place_sell_order(self, orderbook_id: str, price: float, volume: int):
        order_id = self._next_id("SELL")
        self.placed_sells.append({"ob_id": orderbook_id, "price": price,
                                  "qty": volume, "order_id": order_id})
        self.open_orders.append({"orderId": order_id, "orderbookId": orderbook_id,
                                 "side": "SELL", "price": price, "volume": volume})
        return {"orderRequestStatus": "SUCCESS", "orderId": order_id}

    def place_stop_loss(self, orderbook_id: str, trigger_price: float,
                        sell_price: float, volume: int):
        order_id = self._next_id("STOP")
        self.placed_stops.append({"ob_id": orderbook_id,
                                  "trigger": trigger_price,
                                  "sell": sell_price, "qty": volume,
                                  "order_id": order_id})
        return {"orderRequestStatus": "SUCCESS", "orderId": order_id}

    def cancel_order(self, order_id: str):
        self.cancelled.append(order_id)
        self.open_orders = [o for o in self.open_orders
                            if str(o.get("orderId")) != str(order_id)]
        return {"orderRequestStatus": "SUCCESS"}

    def get_open_orders(self):
        return list(self.open_orders)

    def get_positions(self):
        return list(self.positions)

    def get_quote(self, orderbook_id: str):
        return {"buy": 42.50, "sell": 42.55, "last": 42.52}

    # -- helpers for tests to model fills ----------------------------------

    def fill_order(self, order_id: str, *, position_delta: int = 0,
                   ob_id: str = "") -> None:
        """Simulate Avanza filling an order: remove from open_orders and
        adjust position volume."""
        self.open_orders = [o for o in self.open_orders
                            if str(o.get("orderId")) != str(order_id)]
        if position_delta == 0:
            return
        for p in self.positions:
            if str(p.get("orderbook_id")) == str(ob_id):
                p["volume"] = int(p["volume"]) + position_delta
                return
        if position_delta > 0:
            self.positions.append({"orderbook_id": ob_id,
                                   "volume": position_delta})


CATALOG = {
    "BULL_SILVER_X5_AVA_4": {
        "ob_id": "1650161",
        "underlying": "XAG-USD",
        "direction": "LONG",
        "leverage": 5.0,
        "barrier": 0,
    },
    "BEAR_SILVER_X5_AVA_12": {
        "ob_id": "2286417",
        "underlying": "XAG-USD",
        "direction": "SHORT",
        "leverage": 5.0,
        "barrier": 0,
    },
}


@pytest.fixture
def fake_session():
    return FakeSession()


@pytest.fixture
def fisher(fake_session, tmp_path, monkeypatch):
    # Disable inter-order sleep so tests are instant.
    monkeypatch.setattr(gf, "GRID_DIRECTION_FLIP_COOLDOWN_MIN", 30, raising=True)
    f = GridFisher(
        session=fake_session,
        catalog=CATALOG,
        state_path=tmp_path / "state.json",
        decisions_path=tmp_path / "decisions.jsonl",
    )
    f._order_delay_s = 0.0
    return f


# ---------------------------------------------------------------------------
# Reconcile — fill detection
# ---------------------------------------------------------------------------


class TestReconcileFills:
    def test_buy_marked_filled_when_position_increases(self):
        state = gf.GridFisherState(session_id="2026-05-11")
        inst = InstrumentState(ob_id="1650161", ticker="XAG-USD",
                               cert_name="BULL", active_direction="LONG")
        inst.buy_ladder.append(TierOrder(
            tier=0, order_id="BUY1", price=42.50, qty=24,
            placed_ts="t", status=ORDER_ARMED,
        ))
        state.by_instrument["1650161"] = inst
        positions = [{"orderbook_id": "1650161", "volume": 24}]
        res = reconcile_against_live(state, open_order_ids=set(),
                                     positions=positions)
        assert res.filled_buys == [("1650161", 0, 42.50)]
        assert inst.buy_ladder[0].status == ORDER_FILLED
        assert inst.inventory_units == 24

    def test_buy_marked_cancelled_when_position_unchanged(self):
        state = gf.GridFisherState(session_id="2026-05-11")
        inst = InstrumentState(ob_id="1650161", ticker="XAG-USD",
                               cert_name="BULL", active_direction="LONG")
        inst.buy_ladder.append(TierOrder(
            tier=0, order_id="BUY1", price=42.50, qty=24,
            placed_ts="t", status=ORDER_ARMED,
        ))
        state.by_instrument["1650161"] = inst
        positions = []
        res = reconcile_against_live(state, open_order_ids=set(),
                                     positions=positions)
        assert res.cancelled_buys == [("1650161", 0)]
        assert inst.buy_ladder[0].status == ORDER_CANCELLED
        assert inst.inventory_units == 0

    def test_armed_buy_in_live_kept_armed(self):
        state = gf.GridFisherState(session_id="2026-05-11")
        inst = InstrumentState(ob_id="1650161", ticker="XAG-USD",
                               cert_name="BULL", active_direction="LONG")
        inst.buy_ladder.append(TierOrder(
            tier=0, order_id="BUY1", price=42.50, qty=24,
            placed_ts="t", status=ORDER_ARMED,
        ))
        state.by_instrument["1650161"] = inst
        res = reconcile_against_live(state, open_order_ids={"BUY1"},
                                     positions=[])
        assert res.filled_buys == []
        assert res.cancelled_buys == []
        assert inst.buy_ladder[0].status == ORDER_ARMED

    def test_sell_marked_filled_when_position_drops(self):
        state = gf.GridFisherState(session_id="2026-05-11")
        inst = InstrumentState(ob_id="1650161", ticker="XAG-USD",
                               cert_name="BULL", active_direction="LONG")
        inst.inventory_units = 24
        inst.avg_entry_price = 42.50
        inst.sell_ladder.append(TierOrder(
            tier=0, order_id="SELL1", price=43.30, qty=24,
            placed_ts="t", status=ORDER_ARMED,
        ))
        state.by_instrument["1650161"] = inst
        positions = [{"orderbook_id": "1650161", "volume": 0}]
        res = reconcile_against_live(state, open_order_ids=set(),
                                     positions=positions)
        assert res.filled_sells == [("1650161", 0, 43.30)]
        assert inst.inventory_units == 0
        assert inst.session_pnl_sek == pytest.approx(24 * (43.30 - 42.50))

    def test_partial_fill_only_filled_tier_transitions(self):
        state = gf.GridFisherState(session_id="2026-05-11")
        inst = InstrumentState(ob_id="1650161", ticker="XAG-USD",
                               cert_name="BULL", active_direction="LONG")
        inst.buy_ladder.append(TierOrder(
            tier=0, order_id="BUY1", price=42.50, qty=24,
            placed_ts="t", status=ORDER_ARMED,
        ))
        inst.buy_ladder.append(TierOrder(
            tier=1, order_id="BUY2", price=42.20, qty=28,
            placed_ts="t", status=ORDER_ARMED,
        ))
        state.by_instrument["1650161"] = inst
        # Only tier 0 filled (24 units now in position); tier 1 still resting
        positions = [{"orderbook_id": "1650161", "volume": 24}]
        res = reconcile_against_live(state, open_order_ids={"BUY2"},
                                     positions=positions)
        assert res.filled_buys == [("1650161", 0, 42.50)]
        assert inst.buy_ladder[1].status == ORDER_ARMED


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


class TestPlacement:
    def test_places_three_tiers_on_clean_state(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        placed = fisher.place_buy_ladder(inst, bid=42.50)
        assert placed == 3
        assert len(fake_session.placed_buys) == 3
        assert len(inst.buy_ladder) == 3
        # First tier closest to bid
        assert inst.buy_ladder[0].price == pytest.approx(42.37, abs=0.01)

    def test_skips_existing_armed_tiers(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        inst.buy_ladder.append(TierOrder(
            tier=0, order_id="prev", price=42.37, qty=24,
            placed_ts="t", status=ORDER_ARMED,
        ))
        placed = fisher.place_buy_ladder(inst, bid=42.50)
        assert placed == 2
        assert len(fake_session.placed_buys) == 2

    def test_skips_below_min_order(self, fisher, fake_session, monkeypatch):
        # Use a tiny per-leg budget so all tiers fail min-order
        monkeypatch.setattr("portfolio.grid_tiers.GRID_LEG_SEK", 100,
                            raising=False)
        # Need to also patch the imported reference
        import portfolio.grid_tiers as gt
        monkeypatch.setattr(gt, "GRID_LEG_SEK", 100)
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        # Manually re-import after patch — direct call path
        from portfolio.grid_tiers import build_buy_ladder
        tiers = build_buy_ladder(bid=42.50, leg_sek=100)
        assert all(not t.is_active for t in tiers)

    def test_skips_when_in_cooldown(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        inst.cooldown_until = "2999-01-01T00:00:00Z"
        placed = fisher.place_buy_ladder(inst, bid=42.50)
        assert placed == 0
        assert fake_session.placed_buys == []

    def test_skips_when_cap_breached(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        # Force inventory > cap
        inst.inventory_units = 1000
        inst.avg_entry_price = 1000.0
        placed = fisher.place_buy_ladder(inst, bid=42.50)
        assert placed == 0

    def test_skips_when_session_loss_breached(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        inst.session_pnl_sek = -10_000
        placed = fisher.place_buy_ladder(inst, bid=42.50)
        assert placed == 0

    def test_rate_limit_caps_placements(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        fisher._max_orders_per_min = 2
        placed = fisher.place_buy_ladder(inst, bid=42.50)
        assert placed == 2

    def test_session_error_logged_not_raised(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        fake_session.fail_on_buy = True
        placed = fisher.place_buy_ladder(inst, bid=42.50)
        assert placed == 0
        # No state corruption — no orders recorded
        assert inst.buy_ladder == []

    def test_avanza_rejection_logged(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        fake_session.reject_buy = True
        placed = fisher.place_buy_ladder(inst, bid=42.50)
        assert placed == 0
        assert inst.buy_ladder == []


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


class TestRotation:
    def test_rotate_places_sell_and_stop(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        # Simulate: buy placed, then filled
        inst.buy_ladder.append(TierOrder(
            tier=0, order_id="BUY1", price=42.50, qty=24,
            placed_ts="t", status=ORDER_FILLED, fill_price=42.50,
        ))
        inst.inventory_units = 24
        inst.avg_entry_price = 42.50
        fisher.rotate_on_buy_fill(inst, filled_tier=0)
        assert len(fake_session.placed_sells) == 1
        assert len(fake_session.placed_stops) == 1
        sell = fake_session.placed_sells[0]
        assert sell["price"] == pytest.approx(43.01, abs=0.01)
        stop = fake_session.placed_stops[0]
        assert stop["trigger"] == pytest.approx(41.01, abs=0.01)
        # Sell ladder is now populated
        assert len(inst.sell_ladder) == 1
        assert inst.sell_ladder[0].linked_buy_tier == 0
        # Stop ID tracked
        assert inst.stop_loss_id is not None

    def test_rotate_cancels_old_stop_first(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        inst.stop_loss_id = "OLD_STOP"
        inst.buy_ladder.append(TierOrder(
            tier=0, order_id="BUY1", price=42.50, qty=24,
            placed_ts="t", status=ORDER_FILLED, fill_price=42.50,
        ))
        inst.inventory_units = 24
        inst.avg_entry_price = 42.50
        fisher.rotate_on_buy_fill(inst, filled_tier=0)
        assert "OLD_STOP" in fake_session.cancelled

    def test_rotate_noop_when_no_filled_tier(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        fisher.rotate_on_buy_fill(inst, filled_tier=0)
        assert fake_session.placed_sells == []


# ---------------------------------------------------------------------------
# Direction flip
# ---------------------------------------------------------------------------


class TestDirectionFlip:
    def test_flip_cancels_armed_buys(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        # Place a couple of buys first
        fisher.place_buy_ladder(inst, bid=42.50)
        assert len(inst.buy_ladder) == 3
        fisher.arm_direction(inst, "SHORT")
        # All previous buys cancelled, fresh state ready for SHORT
        assert len(inst.buy_ladder) == 0
        assert len(fake_session.cancelled) == 3
        assert inst.active_direction == "SHORT"
        assert inst.cooldown_until is not None

    def test_flip_no_op_when_direction_unchanged(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        fisher.place_buy_ladder(inst, bid=42.50)
        ladder_before = list(inst.buy_ladder)
        fisher.arm_direction(inst, "LONG")
        # Same direction => no cancellation
        assert inst.buy_ladder == ladder_before
        assert fake_session.cancelled == []


# ---------------------------------------------------------------------------
# Probe mode
# ---------------------------------------------------------------------------


class TestProbeOnly:
    def test_probe_records_decisions_without_avanza_calls(self, fisher,
                                                          fake_session):
        fisher._probe_only = True
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        placed = fisher.place_buy_ladder(inst, bid=42.50)
        assert placed == 3
        assert fake_session.placed_buys == []
        # Decisions land in the log though
        log_text = (fisher.decisions_path).read_text()
        assert "probe_placement" in log_text


# ---------------------------------------------------------------------------
# Cancellation helper
# ---------------------------------------------------------------------------


class TestCancelArmedBuys:
    def test_cancels_via_session(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        fisher.place_buy_ladder(inst, bid=42.50)
        n = fisher.cancel_armed_buys(inst)
        assert n == 3
        assert len(fake_session.cancelled) == 3
        assert all(t.status == ORDER_CANCELLED for t in inst.buy_ladder)


# ---------------------------------------------------------------------------
# tick() — main entry
# ---------------------------------------------------------------------------


class TestTick:
    def test_skips_when_disabled(self, fisher, fake_session):
        fisher._enabled = False
        report = fisher.tick(signal_data={"XAG-USD": {"direction": "LONG",
                                                       "confidence": 0.7}})
        assert report.get("skipped_reason") == "disabled"
        assert fake_session.placed_buys == []

    def test_skips_instrument_without_signal(self, fisher, fake_session):
        # XAU-USD direction missing entirely
        report = fisher.tick(signal_data={"XAG-USD": {"direction": "LONG",
                                                       "confidence": 0.7}})
        # XAG placed; the others skipped
        assert report["placements"] == 3
        # Check the instrument reports include no_direction for XAU-USD
        # (and OIL-USD when in catalog).
        all_skips = [v.get("skip") for v in report["instruments"].values()
                     if v.get("ticker") != "XAG-USD"]
        assert any(s == "no_direction" for s in all_skips)

    def test_below_min_confidence_skipped(self, fisher, fake_session):
        report = fisher.tick(signal_data={"XAG-USD": {"direction": "LONG",
                                                       "confidence": 0.3}})
        assert report["placements"] == 0
        skip = next(v["skip"] for v in report["instruments"].values()
                    if v.get("ticker") == "XAG-USD")
        assert "low_conf" in skip

    def test_signal_long_arms_bull_only(self, fisher, fake_session):
        # LONG signal → BULL_SILVER armed; BEAR_SILVER skipped
        fisher.tick(signal_data={"XAG-USD": {"direction": "LONG",
                                              "confidence": 0.7}})
        bull = fisher.state.by_instrument["1650161"]
        bear = fisher.state.by_instrument["2286417"]
        assert bull.active_direction == "LONG"
        assert bear.active_direction == "SHORT"
        assert len(bull.armed_buy_tiers()) > 0
        assert len(bear.armed_buy_tiers()) == 0

    def test_signal_flip_cancels_old_side(self, fisher, fake_session):
        # First tick: LONG → BULL fills its ladder
        fisher.tick(signal_data={"XAG-USD": {"direction": "LONG",
                                              "confidence": 0.7}})
        bull = fisher.state.by_instrument["1650161"]
        assert len(bull.armed_buy_tiers()) == 3
        n_cancels_before = len(fake_session.cancelled)
        # Second tick: signal flips SHORT → BULL's armed buys cancelled,
        # BEAR starts placing.
        fisher.tick(signal_data={"XAG-USD": {"direction": "SHORT",
                                              "confidence": 0.7}})
        # BULL's buys cancelled — 3 new cancellations against Avanza.
        assert len(fake_session.cancelled) >= n_cancels_before + 3
        # BEAR is now placing.
        bear = fisher.state.by_instrument["2286417"]
        assert len(bear.armed_buy_tiers()) == 3

    def test_global_halt_persists(self, fisher, fake_session, monkeypatch):
        # Make halt trigger on tiny loss
        monkeypatch.setattr(gf, "GRID_PER_SESSION_LOSS_LIMIT_SEK", 1)
        fisher.state.global_session_pnl_sek = -100
        report = fisher.tick(signal_data={})
        assert report.get("halted")
        assert fisher.state.halted

    def test_eod_sweep_cancels_buys_close_to_close(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        fisher.place_buy_ladder(inst, bid=42.50)
        # Sweep window: 8 minutes before close
        fisher.tick(signal_data={}, eod_minutes_remaining=8.0)
        # Buys all cancelled via session
        assert len(fake_session.cancelled) == 3

    def test_eod_market_flat_at_zero_minutes(self, fisher, fake_session):
        inst = fisher.state.by_instrument["1650161"]
        inst.active_direction = "LONG"
        inst.inventory_units = 24
        inst.avg_entry_price = 42.50
        fisher.tick(signal_data={}, eod_minutes_remaining=2.0)
        # Aggressive sell placed for inventory
        sell_calls = [s for s in fake_session.placed_sells
                      if s["ob_id"] == "1650161"]
        assert len(sell_calls) >= 1


# ---------------------------------------------------------------------------
# minutes_until_eod
# ---------------------------------------------------------------------------


class TestMinutesUntilEod:
    def test_returns_finite_minutes(self):
        import datetime as _dt
        now = _dt.datetime(2026, 5, 11, 12, 0, 0, tzinfo=_dt.timezone.utc)
        mins = gf.minutes_until_eod(now)
        # 14:00 Stockholm in May (DST = CEST = UTC+2), cutoff at 21:55.
        # Expected window ~ 7h55m (475 min) on a CEST day.
        assert 400 < mins < 600

    def test_after_cutoff_rolls_to_next_day(self):
        import datetime as _dt
        # 22:00 Stockholm CEST = 20:00 UTC. Cutoff already passed.
        now = _dt.datetime(2026, 5, 11, 20, 30, 0, tzinfo=_dt.timezone.utc)
        mins = gf.minutes_until_eod(now)
        # > 23 hours away
        assert mins > 60 * 23
