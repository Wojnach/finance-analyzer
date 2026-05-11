"""State-machine tests for portfolio.grid_fisher.

Covers schema serialization, atomic persistence, session rolling,
direction-flip logic, fill recording (buy + sell weighted avg + P&L),
and halt-detection helpers. No Avanza side effects.
"""

from __future__ import annotations

import json

import pytest

from portfolio import grid_fisher as gf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_state():
    return gf.GridFisherState(session_id="2026-05-11")


@pytest.fixture
def silver_long_inst():
    return gf.InstrumentState(
        ob_id="1650161",
        ticker="XAG-USD",
        cert_name="BULL_SILVER_X5_AVA_4",
        active_direction="LONG",
    )


def _make_armed_buy(tier: int, price: float, qty: int, order_id: str = "buy"):
    return gf.TierOrder(
        tier=tier,
        order_id=order_id,
        price=price,
        qty=qty,
        placed_ts="2026-05-11T10:00:00Z",
        status=gf.ORDER_ARMED,
    )


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


class TestSchemaRoundTrip:
    def test_tier_order_roundtrip(self):
        t = gf.TierOrder(
            tier=2,
            order_id="abc",
            price=42.50,
            qty=24,
            placed_ts="2026-05-11T10:00:00Z",
            status=gf.ORDER_FILLED,
            fill_ts="2026-05-11T10:05:00Z",
            fill_price=42.48,
            p_fill_session=0.65,
        )
        rt = gf.TierOrder.from_dict(t.to_dict())
        assert rt == t

    def test_instrument_state_roundtrip(self, silver_long_inst):
        silver_long_inst.buy_ladder.append(_make_armed_buy(0, 42.50, 24))
        silver_long_inst.inventory_units = 12
        silver_long_inst.avg_entry_price = 42.45
        rt = gf.InstrumentState.from_dict(silver_long_inst.to_dict())
        assert rt.to_dict() == silver_long_inst.to_dict()

    def test_grid_state_roundtrip(self, empty_state, silver_long_inst):
        empty_state.by_instrument[silver_long_inst.ob_id] = silver_long_inst
        rt = gf.GridFisherState.from_dict(empty_state.to_dict())
        assert rt.to_dict() == empty_state.to_dict()


# ---------------------------------------------------------------------------
# Persistence (atomic JSON)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_then_load(self, tmp_path, silver_long_inst):
        state = gf.GridFisherState(session_id="2026-05-11")
        state.by_instrument[silver_long_inst.ob_id] = silver_long_inst
        path = tmp_path / "state.json"
        gf.save_state(state, path)
        loaded = gf.load_state(path)
        assert loaded.to_dict() == state.to_dict()

    def test_load_missing_returns_fresh_state(self, tmp_path):
        loaded = gf.load_state(tmp_path / "nonexistent.json")
        assert loaded.version == gf.GRID_STATE_SCHEMA_VERSION
        assert loaded.by_instrument == {}
        assert loaded.session_id  # populated to today

    def test_load_corrupt_returns_fresh_state(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        loaded = gf.load_state(path)
        assert loaded.by_instrument == {}

    def test_load_old_version_resets(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"version": 0, "session_id": "x"}))
        loaded = gf.load_state(path)
        assert loaded.version == gf.GRID_STATE_SCHEMA_VERSION
        assert loaded.by_instrument == {}

    def test_load_malformed_inner_resets(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text(json.dumps({
            "version": gf.GRID_STATE_SCHEMA_VERSION,
            "by_instrument": {"x": {"ticker": "XAG", "buy_ladder": [{"missing": "fields"}]}},
        }))
        loaded = gf.load_state(path)
        # Fell through to fresh state — by_instrument is the empty seed.
        assert loaded.by_instrument == {}


# ---------------------------------------------------------------------------
# Seed / catalog wiring
# ---------------------------------------------------------------------------


class TestSeed:
    def test_seed_creates_records_for_each_active_instrument(self, empty_state):
        catalog = {
            "BULL_SILVER_X5_AVA_4": {"ob_id": "1650161"},
            "BEAR_SILVER_X5_AVA_12": {"ob_id": "2286417"},
            "BULL_GULD_X5_AVA": {"ob_id": "738811"},
            "BEAR_GULD_X5_VON4": {"ob_id": "1047859"},
            "BULL_OLJAB_X5_AVA_2": {"ob_id": "2367797"},
            "BEAR_OLJAB_X5_AVA_2": {"ob_id": "2367803"},
        }
        gf._seed_state_for_active_instruments(empty_state, catalog)
        # 3 tickers × 2 directions = 6 records
        assert len(empty_state.by_instrument) == 6
        assert "1650161" in empty_state.by_instrument
        assert empty_state.by_instrument["1650161"].cert_name == "BULL_SILVER_X5_AVA_4"

    def test_seed_idempotent(self, empty_state):
        catalog = {"BULL_SILVER_X5_AVA_4": {"ob_id": "1650161"}}
        gf._seed_state_for_active_instruments(empty_state, catalog)
        first = empty_state.to_dict()
        gf._seed_state_for_active_instruments(empty_state, catalog)
        assert empty_state.to_dict() == first


# ---------------------------------------------------------------------------
# Session rolling
# ---------------------------------------------------------------------------


class TestSessionRoll:
    def test_no_roll_when_same_day(self, monkeypatch):
        monkeypatch.setattr(gf, "_today_session_id", lambda: "2026-05-11")
        state = gf.GridFisherState(session_id="2026-05-11")
        state.global_session_pnl_sek = -150.0
        rolled = gf.roll_session_if_new_day(state)
        assert not rolled
        assert state.global_session_pnl_sek == -150.0

    def test_roll_resets_session_counters(self, monkeypatch, silver_long_inst):
        monkeypatch.setattr(gf, "_today_session_id", lambda: "2026-05-12")
        state = gf.GridFisherState(session_id="2026-05-11")
        state.global_session_pnl_sek = -150.0
        silver_long_inst.session_pnl_sek = -50.0
        silver_long_inst.fills_this_session = 3
        state.by_instrument[silver_long_inst.ob_id] = silver_long_inst
        rolled = gf.roll_session_if_new_day(state)
        assert rolled
        assert state.session_id == "2026-05-12"
        assert state.global_session_pnl_sek == 0.0
        assert silver_long_inst.session_pnl_sek == 0.0
        assert silver_long_inst.fills_this_session == 0

    def test_roll_preserves_inventory(self, monkeypatch, silver_long_inst):
        monkeypatch.setattr(gf, "_today_session_id", lambda: "2026-05-12")
        state = gf.GridFisherState(session_id="2026-05-11")
        silver_long_inst.inventory_units = 24
        silver_long_inst.avg_entry_price = 42.50
        state.by_instrument[silver_long_inst.ob_id] = silver_long_inst
        gf.roll_session_if_new_day(state)
        assert silver_long_inst.inventory_units == 24
        assert silver_long_inst.avg_entry_price == 42.50


# ---------------------------------------------------------------------------
# Direction flip
# ---------------------------------------------------------------------------


class TestFlipDirection:
    def test_flip_clears_buy_ladder_and_sets_cooldown(self, silver_long_inst):
        silver_long_inst.buy_ladder.append(_make_armed_buy(0, 42.50, 24))
        gf.flip_direction(silver_long_inst, "SHORT", cooldown_min=30)
        assert silver_long_inst.active_direction == "SHORT"
        assert silver_long_inst.buy_ladder == []
        assert silver_long_inst.cooldown_until is not None
        assert silver_long_inst.last_direction_flip_ts is not None

    def test_flip_preserves_sell_ladder_and_inventory(self, silver_long_inst):
        silver_long_inst.sell_ladder.append(
            gf.TierOrder(tier=0, order_id="s", price=43.30, qty=24,
                         placed_ts="t", linked_buy_tier=0)
        )
        silver_long_inst.inventory_units = 24
        gf.flip_direction(silver_long_inst, "SHORT", cooldown_min=30)
        assert silver_long_inst.inventory_units == 24
        assert len(silver_long_inst.sell_ladder) == 1

    def test_flip_to_same_direction_noop(self, silver_long_inst):
        before = silver_long_inst.to_dict()
        gf.flip_direction(silver_long_inst, "LONG")
        # Identical except for transient bookkeeping
        assert silver_long_inst.active_direction == "LONG"
        assert silver_long_inst.cooldown_until == before["cooldown_until"]

    def test_flip_rejects_invalid_direction(self, silver_long_inst):
        with pytest.raises(ValueError, match="direction"):
            gf.flip_direction(silver_long_inst, "BOTH")


# ---------------------------------------------------------------------------
# Record fill (the meat of the state machine)
# ---------------------------------------------------------------------------


class TestRecordFill:
    def test_buy_fill_updates_inventory_and_avg(self, silver_long_inst):
        silver_long_inst.buy_ladder.append(_make_armed_buy(0, 42.50, 24))
        order = gf.record_fill(silver_long_inst, tier_idx=0, fill_price=42.48,
                               side="buy")
        assert order is not None
        assert order.status == gf.ORDER_FILLED
        assert order.fill_price == 42.48
        assert silver_long_inst.inventory_units == 24
        assert silver_long_inst.avg_entry_price == pytest.approx(42.48)
        assert silver_long_inst.fills_this_session == 1

    def test_second_buy_weighted_average(self, silver_long_inst):
        silver_long_inst.buy_ladder.append(_make_armed_buy(0, 42.50, 24))
        silver_long_inst.buy_ladder.append(_make_armed_buy(1, 42.20, 28))
        gf.record_fill(silver_long_inst, tier_idx=0, fill_price=42.48, side="buy")
        gf.record_fill(silver_long_inst, tier_idx=1, fill_price=42.20, side="buy")
        assert silver_long_inst.inventory_units == 52
        expected = (42.48 * 24 + 42.20 * 28) / 52
        assert silver_long_inst.avg_entry_price == pytest.approx(expected)

    def test_sell_fill_realises_pnl_and_reduces_inventory(self, silver_long_inst):
        silver_long_inst.inventory_units = 24
        silver_long_inst.avg_entry_price = 42.50
        silver_long_inst.sell_ladder.append(
            gf.TierOrder(tier=0, order_id="s", price=43.30, qty=24,
                         placed_ts="t", linked_buy_tier=0)
        )
        gf.record_fill(silver_long_inst, tier_idx=0, fill_price=43.30, side="sell")
        assert silver_long_inst.inventory_units == 0
        assert silver_long_inst.avg_entry_price == 0.0
        # 24 * (43.30 - 42.50) = 19.20
        assert silver_long_inst.session_pnl_sek == pytest.approx(19.20)
        assert silver_long_inst.consecutive_losses == 0

    def test_losing_sell_increments_consecutive_losses(self, silver_long_inst):
        silver_long_inst.inventory_units = 24
        silver_long_inst.avg_entry_price = 42.50
        silver_long_inst.sell_ladder.append(
            gf.TierOrder(tier=0, order_id="s", price=41.00, qty=24, placed_ts="t")
        )
        gf.record_fill(silver_long_inst, tier_idx=0, fill_price=41.00, side="sell")
        assert silver_long_inst.session_pnl_sek == pytest.approx(-36.00)
        assert silver_long_inst.consecutive_losses == 1

    def test_no_armed_tier_returns_none(self, silver_long_inst):
        assert gf.record_fill(silver_long_inst, tier_idx=0,
                              fill_price=42.0, side="buy") is None

    def test_invalid_side_raises(self, silver_long_inst):
        with pytest.raises(ValueError, match="side"):
            gf.record_fill(silver_long_inst, tier_idx=0, fill_price=42.0,
                           side="flat")


# ---------------------------------------------------------------------------
# Cancellation + pruning
# ---------------------------------------------------------------------------


class TestCancelAndPrune:
    def test_cancel_marks_armed_tier(self, silver_long_inst):
        silver_long_inst.buy_ladder.append(_make_armed_buy(0, 42.50, 24))
        cancelled = gf.cancel_buy_tier(silver_long_inst, tier_idx=0)
        assert cancelled is not None
        assert cancelled.status == gf.ORDER_CANCELLED

    def test_cancel_unknown_returns_none(self, silver_long_inst):
        assert gf.cancel_buy_tier(silver_long_inst, tier_idx=9) is None

    def test_prune_drops_terminal_orders(self, silver_long_inst):
        silver_long_inst.buy_ladder.append(_make_armed_buy(0, 42.50, 24))
        silver_long_inst.buy_ladder.append(_make_armed_buy(1, 42.20, 28))
        gf.record_fill(silver_long_inst, tier_idx=0, fill_price=42.48, side="buy")
        gf.cancel_buy_tier(silver_long_inst, tier_idx=1)
        gf.prune_terminal_orders(silver_long_inst)
        assert silver_long_inst.buy_ladder == []


# ---------------------------------------------------------------------------
# Health / safety
# ---------------------------------------------------------------------------


class TestHealth:
    def test_per_instrument_cap_detects_breach(self, silver_long_inst, monkeypatch):
        monkeypatch.setattr(gf, "GRID_PER_INSTRUMENT_MAX_SEK", 1000)
        silver_long_inst.inventory_units = 30
        silver_long_inst.avg_entry_price = 40.0  # 1200 SEK held
        assert silver_long_inst.hit_per_instrument_cap()

    def test_session_loss_breached_detects(self, silver_long_inst, monkeypatch):
        monkeypatch.setattr(gf, "GRID_PER_SESSION_LOSS_LIMIT_SEK", 500)
        silver_long_inst.session_pnl_sek = -510.0
        assert silver_long_inst.session_loss_breached()
        silver_long_inst.session_pnl_sek = -490.0
        assert not silver_long_inst.session_loss_breached()

    def test_cooldown_check(self, silver_long_inst):
        silver_long_inst.cooldown_until = "2026-05-11T10:00:00Z"
        assert silver_long_inst.in_cooldown(now_iso="2026-05-11T09:30:00Z")
        assert not silver_long_inst.in_cooldown(now_iso="2026-05-11T10:30:00Z")
        silver_long_inst.cooldown_until = None
        assert not silver_long_inst.in_cooldown(now_iso="2026-05-11T10:30:00Z")

    def test_should_halt_global_at_threshold(self, empty_state, silver_long_inst,
                                             monkeypatch):
        monkeypatch.setattr(gf, "GRID_PER_SESSION_LOSS_LIMIT_SEK", 500)
        empty_state.by_instrument[silver_long_inst.ob_id] = silver_long_inst
        empty_state.global_session_pnl_sek = -250.0
        assert gf.should_halt_global(empty_state) is None
        empty_state.global_session_pnl_sek = -600.0  # > 500 * 1 instrument
        assert gf.should_halt_global(empty_state) is not None


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------


class TestSummarise:
    def test_summary_includes_per_instrument_block(self, empty_state, silver_long_inst):
        silver_long_inst.inventory_units = 24
        silver_long_inst.avg_entry_price = 42.50
        silver_long_inst.session_pnl_sek = 19.20
        empty_state.by_instrument[silver_long_inst.ob_id] = silver_long_inst
        s = gf.summarise(empty_state)
        block = s["by_instrument"]["1650161"]
        assert block["ticker"] == "XAG-USD"
        assert block["inventory_units"] == 24
        assert block["session_pnl_sek"] == 19.2

    def test_summary_excludes_raw_ladders(self, empty_state, silver_long_inst):
        silver_long_inst.buy_ladder.append(_make_armed_buy(0, 42.50, 24))
        empty_state.by_instrument[silver_long_inst.ob_id] = silver_long_inst
        s = gf.summarise(empty_state)
        block = s["by_instrument"]["1650161"]
        assert "buy_ladder" not in block
        assert block["armed_buys"] == 1


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------


class TestDecisionLog:
    def test_log_decision_appends_jsonl(self, tmp_path):
        path = tmp_path / "decisions.jsonl"
        gf.log_decision("placement", ob_id="x", ticker="XAG-USD",
                        decisions_path=path,
                        tier=0, price=42.50, qty=24)
        gf.log_decision("fill", ob_id="x", ticker="XAG-USD",
                        decisions_path=path,
                        tier=0, fill_price=42.48)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["category"] == "placement"
        assert first["tier"] == 0
        assert first["ob_id"] == "x"
        assert "ts" in first
