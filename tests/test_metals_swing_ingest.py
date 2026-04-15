"""Unit tests for SwingTrader.ingest_position and _migrate_orphans.

2026-04-15: added so orphan Avanza positions (e.g. bull_silver_x5 that
peaked +5.78% today with no trailing stop) get adopted into swing
management at startup. The problem was that positions held on Avanza but
not opened via SwingTrader's _execute_buy path were invisible to the
exit machinery. See docs/PLAN-orphan-positions.md for the design.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import patch

import pytest

# data/ contains script-style modules, not a package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import metals_swing_trader as mst


def _make_trader(tmp_path, cash: float = 50_000.0):
    """Minimal SwingTrader bypassing __init__ — sized for ingestion tests."""
    trader = mst.SwingTrader.__new__(mst.SwingTrader)
    trader.page = object()
    trader.state = mst._default_state()
    trader.state["cash_sek"] = cash
    trader.check_count = 0
    trader.regime_history = {}
    trader.cash_sync_ok = True
    trader.cash_sync_was_ok = True
    trader.recon_failure_streak = 0
    trader.reconciled_once = True
    trader._jit_sync_tick = -1
    trader.kelly_no_edge_count = {}
    trader._orphans_migrated = False
    trader.warrant_catalog = {
        "MINI_L_SILVER_AVA_331": {
            "ob_id": "2379768",
            "api_type": "warrant",
            "underlying": "XAG-USD",
            "direction": "LONG",
            "leverage": 5.0,
            "name": "MINI L SILVER AVA 331",
        },
    }
    return trader


def _known_warrants_fixture():
    """Shape mirrors KNOWN_WARRANT_OB_IDS in data/metals_loop.py."""
    return {
        "1650161": {
            "key": "bull_silver_x5",
            "name": "BULL SILVER X5 AVA 4",
            "api_type": "certificate",
            "underlying": "XAG-USD",
            "leverage": 5.0,
            "_managed_by": "swing_trader",
        },
        "2286417": {
            "key": "bear_silver_x5",
            "name": "BEAR SILVER X5 AVA 12",
            "api_type": "certificate",
            "underlying": "XAG-USD",
            "leverage": 5.0,
            "_managed_by": "swing_trader",
        },
        "856394": {
            "key": "gold",
            "name": "BULL GULD X8 N",
            "api_type": "certificate",
            "underlying": "XAU-USD",
            "leverage": 8.0,
            "_managed_by": "swing_trader",
        },
    }


@pytest.fixture
def patched_legacy_state(tmp_path, monkeypatch):
    """Point ingest_position's legacy-state writer at a tmp file (xdist safe)."""
    legacy_path = tmp_path / "metals_positions_state.json"
    # Pre-seed with an active bull_silver_x5 entry (the real-world case).
    import json
    legacy_path.write_text(
        json.dumps(
            {
                "bull_silver_x5": {
                    "active": True,
                    "units": 97,
                    "entry": 10.27,
                    "stop": 9.76,
                    "ob_id": "1650161",
                },
            },
            indent=2,
        )
    )
    monkeypatch.setattr(mst, "LEGACY_POSITIONS_FILE", str(legacy_path), raising=False)
    return legacy_path


@pytest.fixture
def patched_state_file(tmp_path, monkeypatch):
    """Redirect STATE_FILE so _save_state doesn't touch the live swing state."""
    state_path = tmp_path / "metals_swing_state.json"
    monkeypatch.setattr(mst, "STATE_FILE", str(state_path))
    return state_path


# ---------------------------------------------------------------------------
# ingest_position
# ---------------------------------------------------------------------------


def test_ingest_creates_pos_with_all_required_fields(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)

    pos_id = trader.ingest_position(
        ob_id="2379768",
        units=97,
        entry_price=14.70,
        underlying_price=79.42,
        direction="LONG",
        set_stop_loss=False,
    )

    assert pos_id is not None
    assert pos_id.startswith("pos_")
    pos = trader.state["positions"][pos_id]

    for field in (
        "warrant_key", "warrant_name", "ob_id", "api_type", "underlying",
        "direction", "units", "entry_price", "entry_underlying", "entry_ts",
        "peak_underlying", "trough_underlying", "trailing_active",
        "stop_order_id", "leverage", "fill_verified", "buy_order_id",
    ):
        assert field in pos, f"missing field {field}"

    assert pos["ob_id"] == "2379768"
    assert pos["units"] == 97
    assert pos["entry_price"] == 14.70
    assert pos["entry_underlying"] == 79.42
    assert pos["peak_underlying"] == 79.42
    assert pos["trough_underlying"] == 79.42
    assert pos["direction"] == "LONG"
    assert pos["trailing_active"] is False
    assert pos["stop_order_id"] is None
    assert pos["fill_verified"] is True
    assert pos["buy_order_id"] is None
    assert pos["ingested"] is True


def test_ingest_rejects_duplicate_ob_id(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)

    trader.state["positions"]["pos_existing"] = {
        "ob_id": "2379768", "warrant_name": "MINI L SILVER AVA 331", "units": 50,
    }

    pos_id = trader.ingest_position(
        ob_id="2379768",
        units=97,
        entry_price=14.70,
        underlying_price=79.42,
        set_stop_loss=False,
    )

    assert pos_id is None
    assert len(trader.state["positions"]) == 1  # did not add duplicate


def test_ingest_looks_up_catalog_by_ob_id_via_swing_catalog(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)

    pos_id = trader.ingest_position(
        ob_id="2379768",  # MINI L SILVER AVA 331 in the swing catalog
        units=97,
        entry_price=14.70,
        underlying_price=79.42,
        set_stop_loss=False,
    )

    assert pos_id is not None
    pos = trader.state["positions"][pos_id]
    assert pos["warrant_key"] == "MINI_L_SILVER_AVA_331"
    assert pos["underlying"] == "XAG-USD"
    assert pos["leverage"] == 5.0


def test_ingest_falls_back_to_known_warrants(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    """If the swing catalog doesn't have the ob_id, fall back to KNOWN_WARRANT_OB_IDS."""
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    # Simulate a hardcoded bull_silver_x5 known only via metals_loop.
    monkeypatch.setattr(
        mst, "_lookup_known_warrant", lambda ob: _known_warrants_fixture().get(str(ob)),
        raising=False,
    )

    pos_id = trader.ingest_position(
        ob_id="1650161",
        units=97,
        entry_price=10.27,
        underlying_price=78.95,
        set_stop_loss=False,
    )

    assert pos_id is not None
    pos = trader.state["positions"][pos_id]
    assert pos["warrant_key"] == "bull_silver_x5"
    assert pos["warrant_name"] == "BULL SILVER X5 AVA 4"
    assert pos["leverage"] == 5.0
    assert pos["api_type"] == "certificate"


def test_ingest_skips_unknown_ob_id_and_returns_none(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    monkeypatch.setattr(mst, "_lookup_known_warrant", lambda ob: None, raising=False)

    pos_id = trader.ingest_position(
        ob_id="9999999",
        units=10,
        entry_price=100.0,
        underlying_price=50.0,
        set_stop_loss=False,
    )

    assert pos_id is None
    assert trader.state["positions"] == {}


def test_ingest_calls_set_stop_loss_when_requested(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", False)
    # No pre-existing Avanza stop → ingest places a new one.
    monkeypatch.setattr(mst, "_find_existing_stop", lambda ob, u: None)

    called = []

    def _capture(pos_id):
        called.append(pos_id)
        trader.state["positions"][pos_id]["stop_order_id"] = "STOP_FAKE_123"

    trader._set_stop_loss = _capture

    pos_id = trader.ingest_position(
        ob_id="2379768",
        units=97,
        entry_price=14.70,
        underlying_price=79.42,
        set_stop_loss=True,
    )

    assert pos_id is not None
    assert called == [pos_id]


def test_ingest_skips_set_stop_loss_on_dry_run(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)

    called = []
    trader._set_stop_loss = lambda pos_id: called.append(pos_id)

    pos_id = trader.ingest_position(
        ob_id="2379768", units=97, entry_price=14.70,
        underlying_price=79.42, set_stop_loss=True,
    )

    # DRY_RUN prevents real stop placement; ingest still creates the pos.
    assert pos_id is not None
    assert called == []


def test_ingest_does_not_decrement_cash(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    trader = _make_trader(tmp_path, cash=12_345.67)
    monkeypatch.setattr(mst, "DRY_RUN", True)

    trader.ingest_position(
        ob_id="2379768", units=97, entry_price=14.70,
        underlying_price=79.42, set_stop_loss=False,
    )

    # Ingestion adopts an existing position; cash was already spent long
    # ago (possibly in a prior process). Do not double-count it now.
    assert trader.state["cash_sek"] == 12_345.67


def test_ingest_writes_trade_record_with_action_INGEST(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)

    logged = []
    monkeypatch.setattr(mst, "_log_trade", lambda rec: logged.append(rec))

    trader.ingest_position(
        ob_id="2379768", units=97, entry_price=14.70,
        underlying_price=79.42, set_stop_loss=False,
    )

    assert len(logged) == 1
    assert logged[0]["action"] == "INGEST"
    assert logged[0]["units"] == 97
    assert logged[0]["warrant_key"] == "MINI_L_SILVER_AVA_331"
    assert logged[0]["ingested"] is True


def test_ingest_marks_legacy_state_inactive(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    """After adoption, the matching legacy entry should be deactivated."""
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    monkeypatch.setattr(
        mst, "_lookup_known_warrant",
        lambda ob: _known_warrants_fixture().get(str(ob)),
        raising=False,
    )

    trader.ingest_position(
        ob_id="1650161", units=97, entry_price=10.27,
        underlying_price=78.95, set_stop_loss=False,
    )

    import json
    legacy = json.loads(patched_legacy_state.read_text())
    assert legacy["bull_silver_x5"]["active"] is False
    assert legacy["bull_silver_x5"]["sold_reason"] == "migrated_to_swing"
    assert "sold_ts" in legacy["bull_silver_x5"]


# ---------------------------------------------------------------------------
# _migrate_orphans
# ---------------------------------------------------------------------------


def test_migrate_orphans_skips_already_tracked_positions(tmp_path, monkeypatch, patched_state_file):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)

    # Both positions already in swing state
    trader.state["positions"]["pos_1"] = {
        "ob_id": "2379768", "warrant_name": "MINI L SILVER AVA 331", "units": 97,
    }

    monkeypatch.setattr(
        mst, "fetch_page_positions",
        lambda page, acct: {"2379768": {
            "name": "MINI L SILVER AVA 331", "units": 97,
            "value": 1426.0, "avg_price": 14.70, "api_type": "warrant",
        }},
    )

    ingested = []
    original = trader.ingest_position
    trader.ingest_position = lambda **kw: ingested.append(kw) or None

    trader._migrate_orphans(prices={"XAG-USD": {"price_usd": 79.42}})

    assert ingested == []


def test_migrate_orphans_ingests_orphan(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    monkeypatch.setattr(
        mst, "_lookup_known_warrant",
        lambda ob: _known_warrants_fixture().get(str(ob)),
        raising=False,
    )
    monkeypatch.setattr(mst, "FISHING_OB_IDS", set(), raising=False)

    # Avanza has bull_silver_x5 but it's NOT in swing state
    monkeypatch.setattr(
        mst, "fetch_page_positions",
        lambda page, acct: {"1650161": {
            "name": "BULL SILVER X5 AVA 4", "units": 97,
            "value": 984.05, "avg_price": 10.27, "api_type": "certificate",
        }},
    )

    # Codex review 2026-04-15 P1: `prices` is keyed by position name
    # (e.g. silver301), not by ticker. Keys containing "silver" map to
    # XAG-USD via _get_ticker_underlying_price's heuristic.
    trader._migrate_orphans(prices={"silver_ref": {"underlying": 78.95}})

    # Position should now be in swing state
    assert any(
        pos["ob_id"] == "1650161"
        for pos in trader.state["positions"].values()
    )


def test_migrate_orphans_uses_fetch_price_fallback_for_underlying(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    """Codex review 2026-04-15 P1: when prices dict lacks the underlying
    (e.g. state-wiped orphan with no legacy POSITIONS entry driving the
    main loop's price fetch), fall back to fetch_price(page, ob_id)."""
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    monkeypatch.setattr(
        mst, "_lookup_known_warrant",
        lambda ob: _known_warrants_fixture().get(str(ob)),
        raising=False,
    )
    monkeypatch.setattr(mst, "FISHING_OB_IDS", set(), raising=False)
    monkeypatch.setattr(
        mst, "fetch_page_positions",
        lambda page, acct: {"1650161": {
            "name": "BULL SILVER X5 AVA 4", "units": 97,
            "value": 984.05, "avg_price": 10.27, "api_type": "certificate",
        }},
    )
    monkeypatch.setattr(
        mst, "fetch_price",
        lambda page, ob, api_type: {"bid": 10.25, "underlying": 78.95},
    )

    # Empty prices dict forces fetch_price fallback
    status = trader._migrate_orphans(prices={})

    assert status == "success"
    assert any(
        pos["ob_id"] == "1650161"
        for pos in trader.state["positions"].values()
    )


def test_migrate_orphans_skips_fishing_ob_ids(tmp_path, monkeypatch, patched_state_file):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    monkeypatch.setattr(
        mst, "_lookup_known_warrant",
        lambda ob: _known_warrants_fixture().get(str(ob)),
        raising=False,
    )
    monkeypatch.setattr(mst, "FISHING_OB_IDS", {"1650161"}, raising=False)

    monkeypatch.setattr(
        mst, "fetch_page_positions",
        lambda page, acct: {"1650161": {
            "name": "BULL SILVER X5 AVA 4", "units": 97,
            "value": 984.05, "avg_price": 10.27, "api_type": "certificate",
        }},
    )

    trader._migrate_orphans(prices={"XAG-USD": {"price_usd": 78.95}})

    # Fishing-flagged ob_id should NOT be ingested
    assert all(
        pos["ob_id"] != "1650161"
        for pos in trader.state["positions"].values()
    )


def test_migrate_orphans_skips_missing_underlying_price(tmp_path, monkeypatch, patched_state_file):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    monkeypatch.setattr(
        mst, "_lookup_known_warrant",
        lambda ob: _known_warrants_fixture().get(str(ob)),
        raising=False,
    )
    monkeypatch.setattr(mst, "FISHING_OB_IDS", set(), raising=False)

    monkeypatch.setattr(
        mst, "fetch_page_positions",
        lambda page, acct: {"1650161": {
            "name": "BULL SILVER X5 AVA 4", "units": 97,
            "value": 984.05, "avg_price": 10.27, "api_type": "certificate",
        }},
    )

    # No XAG-USD in prices → migration should skip (retry next tick)
    trader._migrate_orphans(prices={})

    assert all(
        pos["ob_id"] != "1650161"
        for pos in trader.state["positions"].values()
    )


def test_migrate_orphans_handles_none_from_fetch(tmp_path, monkeypatch, patched_state_file):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    monkeypatch.setattr(mst, "fetch_page_positions", lambda p, a: None)

    # Should not raise; no state change
    status = trader._migrate_orphans(prices={"XAG-USD": {"price_usd": 78.95}})

    assert trader.state["positions"] == {}
    assert status == "partial"


def test_migrate_orphans_returns_partial_when_underlying_missing(tmp_path, monkeypatch, patched_state_file):
    """Codex review 2026-04-15 P1: partial status blocks one-shot disable."""
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    monkeypatch.setattr(
        mst, "_lookup_known_warrant",
        lambda ob: _known_warrants_fixture().get(str(ob)),
        raising=False,
    )
    monkeypatch.setattr(mst, "FISHING_OB_IDS", set(), raising=False)
    # fetch_page_positions returns a known orphan...
    monkeypatch.setattr(
        mst, "fetch_page_positions",
        lambda p, a: {"1650161": {
            "name": "BULL SILVER X5 AVA 4", "units": 97,
            "value": 984.05, "avg_price": 10.27, "api_type": "certificate",
        }},
    )
    # ...but fetch_price fallback also fails, so we truly cannot seed und_price
    monkeypatch.setattr(mst, "fetch_price", lambda p, ob, at: None)

    status = trader._migrate_orphans(prices={})

    assert status == "partial"
    # Nothing adopted because underlying price unavailable
    assert all(pos["ob_id"] != "1650161" for pos in trader.state["positions"].values())


def test_migrate_orphans_returns_success_on_empty_account(tmp_path, monkeypatch, patched_state_file):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "fetch_page_positions", lambda p, a: {})

    status = trader._migrate_orphans(prices={})

    assert status == "success"


def test_migrate_orphans_returns_disabled_when_flag_off(tmp_path, monkeypatch, patched_state_file):
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "SWING_INGEST_ORPHANS", False)

    status = trader._migrate_orphans(prices={})

    assert status == "disabled"


def test_ingest_includes_ob_id_in_pos_id(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    """Codex review 2026-04-15 P2: pos_id must include ob_id to avoid collisions."""
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)

    pos_id = trader.ingest_position(
        ob_id="2379768", units=10, entry_price=14.7,
        underlying_price=79.42, set_stop_loss=False,
    )

    assert pos_id is not None
    assert "2379768" in pos_id


def test_ingest_adopts_existing_avanza_stop_and_skips_placement(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    """Codex review 2026-04-15 P1: reuse existing broker stop instead of placing a duplicate."""
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", False)
    # Pretend Avanza already has a stop on this ob_id
    monkeypatch.setattr(mst, "_find_existing_stop", lambda ob, u: "EXISTING_STOP_999")

    set_stop_calls = []
    trader._set_stop_loss = lambda pid: set_stop_calls.append(pid)

    pos_id = trader.ingest_position(
        ob_id="2379768", units=97, entry_price=14.7,
        underlying_price=79.42, set_stop_loss=True,
    )

    assert pos_id is not None
    # Must NOT call _set_stop_loss when an existing stop was found
    assert set_stop_calls == []
    pos = trader.state["positions"][pos_id]
    assert pos["stop_order_id"] == "EXISTING_STOP_999"
    assert pos["stop_adopted"] is True


def test_find_existing_stop_matches_orderbookid_camelcase(monkeypatch):
    """Codex review round 3 P2: Avanza uses both orderBookId and orderbookId."""
    monkeypatch.setattr(
        "portfolio.avanza_session.get_stop_losses_strict",
        lambda: [
            {"id": "STOP_CAMEL", "orderBookId": "1650161",
             "accountId": mst.ACCOUNT_ID, "volume": 97},
            {"id": "STOP_SNAKE", "orderbookId": "9999999",
             "accountId": mst.ACCOUNT_ID, "volume": 50},
        ],
        raising=False,
    )

    assert mst._find_existing_stop("1650161", 97) == "STOP_CAMEL"
    assert mst._find_existing_stop("9999999", 50) == "STOP_SNAKE"


def test_find_existing_stop_matches_nested_orderbook():
    """Nested {orderbook: {id}} schema."""
    import portfolio.avanza_session
    original = getattr(portfolio.avanza_session, "get_stop_losses_strict", None)
    try:
        portfolio.avanza_session.get_stop_losses_strict = lambda: [
            {"id": "STOP_NESTED", "orderbook": {"id": "1650161"},
             "accountId": mst.ACCOUNT_ID, "volume": 97},
        ]
        assert mst._find_existing_stop("1650161", 97) == "STOP_NESTED"
    finally:
        if original:
            portfolio.avanza_session.get_stop_losses_strict = original


def test_find_existing_stop_reads_volume_from_orderevent(monkeypatch):
    """Codex review round 6 P2: Avanza's orderEvent.volume path."""
    monkeypatch.setattr(
        "portfolio.avanza_session.get_stop_losses_strict",
        lambda: [
            {"id": "STOP_VIA_ORDER_EVENT", "orderBookId": "1650161",
             "accountId": mst.ACCOUNT_ID, "orderEvent": {"volume": 97}},
        ],
        raising=False,
    )
    assert mst._find_existing_stop("1650161", 97) == "STOP_VIA_ORDER_EVENT"


def test_find_existing_stop_rejects_missing_volume(monkeypatch):
    """Codex review round 6 P2: volume ambiguity = don't adopt."""
    monkeypatch.setattr(
        "portfolio.avanza_session.get_stop_losses_strict",
        lambda: [
            {"id": "STOP_NO_VOL", "orderBookId": "1650161",
             "accountId": mst.ACCOUNT_ID},  # no volume anywhere
        ],
        raising=False,
    )
    assert mst._find_existing_stop("1650161", 97) is None


def test_find_existing_stop_returns_sentinel_on_read_failure(monkeypatch):
    """Codex review round 6 P2: fail closed — return _STOP_READ_FAILED."""
    def _raise():
        raise RuntimeError("Avanza API read failed")
    monkeypatch.setattr(
        "portfolio.avanza_session.get_stop_losses_strict", _raise, raising=False,
    )
    result = mst._find_existing_stop("1650161", 97)
    assert result == mst._STOP_READ_FAILED


def test_ingest_defers_stop_placement_when_read_fails(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    """Codex review round 6 P2: on stop-read failure, don't place a new stop."""
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", False)
    monkeypatch.setattr(mst, "_find_existing_stop", lambda ob, u: mst._STOP_READ_FAILED)

    set_stop_calls = []
    trader._set_stop_loss = lambda pid: set_stop_calls.append(pid)

    pos_id = trader.ingest_position(
        ob_id="2379768", units=97, entry_price=14.7,
        underlying_price=79.42, set_stop_loss=True,
    )

    assert pos_id is not None
    # Must NOT place a new stop when we can't verify no pre-existing stop
    assert set_stop_calls == []
    pos = trader.state["positions"][pos_id]
    assert pos["stop_order_id"] is None
    assert "stop_read_failed_ts" in pos


def test_find_existing_stop_returns_none_when_no_match():
    import portfolio.avanza_session
    original = getattr(portfolio.avanza_session, "get_stop_losses_strict", None)
    try:
        portfolio.avanza_session.get_stop_losses_strict = lambda: [
            {"id": "STOP_OTHER", "orderBookId": "9999999"},
        ]
        assert mst._find_existing_stop("1650161", 97) is None
    finally:
        if original:
            portfolio.avanza_session.get_stop_losses_strict = original


def test_find_existing_stop_rejects_wrong_account(monkeypatch):
    """Codex review round 4 P2: don't adopt another account's stop."""
    monkeypatch.setattr(
        "portfolio.avanza_session.get_stop_losses_strict",
        lambda: [
            {"id": "STOP_OTHER_ACCT", "orderBookId": "1650161",
             "accountId": "9999999", "volume": 97},
            {"id": "STOP_OUR_ACCT", "orderBookId": "1650161",
             "accountId": mst.ACCOUNT_ID, "volume": 97},
        ],
        raising=False,
    )
    assert mst._find_existing_stop("1650161", 97) == "STOP_OUR_ACCT"


def test_find_existing_stop_rejects_wrong_volume(monkeypatch):
    """Codex review round 4 P2: don't adopt a partial-volume stop."""
    monkeypatch.setattr(
        "portfolio.avanza_session.get_stop_losses_strict",
        lambda: [
            {"id": "STOP_PARTIAL", "orderBookId": "1650161",
             "accountId": mst.ACCOUNT_ID, "volume": 50},  # covers only 50 of 97
        ],
        raising=False,
    )
    # We hold 97u; a 50u stop must not be adopted
    assert mst._find_existing_stop("1650161", 97) is None


def test_lookup_legacy_underlying_entry_returns_stored_value(tmp_path, monkeypatch):
    """Codex review round 4 P1: preserve the true entry underlying from legacy state."""
    import json
    legacy_path = tmp_path / "metals_positions_state.json"
    legacy_path.write_text(json.dumps({
        "bull_silver_x5": {
            "active": True, "units": 97, "entry": 10.27,
            "stop": 9.76, "ob_id": "1650161",
            "underlying_entry": 78.95,
        },
    }))
    monkeypatch.setattr(mst, "LEGACY_POSITIONS_FILE", str(legacy_path))

    assert mst._lookup_legacy_underlying_entry("1650161") == 78.95
    assert mst._lookup_legacy_underlying_entry("9999999") == 0.0


def test_lookup_legacy_underlying_entry_ignores_migrated_record(tmp_path, monkeypatch):
    """Codex review round 5 P2: don't return stale entry from a migrated (inactive) record."""
    import json
    legacy_path = tmp_path / "metals_positions_state.json"
    legacy_path.write_text(json.dumps({
        "bull_silver_x5": {
            "active": False, "units": 97, "entry": 10.27,
            "stop": 9.76, "ob_id": "1650161",
            "underlying_entry": 78.95,
            "sold_reason": "migrated_to_swing",
            "sold_ts": "2026-04-15T12:00:00+00:00",
        },
    }))
    monkeypatch.setattr(mst, "LEGACY_POSITIONS_FILE", str(legacy_path))

    # Previous trade was migrated; a REBUY of the same ob_id must not
    # reuse the stale entry_underlying.
    assert mst._lookup_legacy_underlying_entry("1650161") == 0.0


def test_migrate_orphans_preserves_true_entry_underlying(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    """Codex review round 4 P1: migration uses legacy underlying_entry when present."""
    # Enrich patched_legacy_state fixture with underlying_entry field
    import json
    legacy_data = json.loads(patched_legacy_state.read_text())
    legacy_data["bull_silver_x5"]["underlying_entry"] = 78.95
    patched_legacy_state.write_text(json.dumps(legacy_data))

    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", True)
    monkeypatch.setattr(
        mst, "_lookup_known_warrant",
        lambda ob: _known_warrants_fixture().get(str(ob)),
        raising=False,
    )
    monkeypatch.setattr(mst, "FISHING_OB_IDS", set(), raising=False)
    monkeypatch.setattr(
        mst, "fetch_page_positions",
        lambda p, a: {"1650161": {
            "name": "BULL SILVER X5 AVA 4", "units": 97,
            "value": 984.05, "avg_price": 10.27, "api_type": "certificate",
        }},
    )

    # Provide a CURRENT spot very different from the entry (80.50 vs 78.95).
    # _migrate_orphans must prefer the legacy stored entry (78.95).
    trader._migrate_orphans(prices={"silver_ref": {"underlying": 80.50}})

    ingested = [p for p in trader.state["positions"].values() if p.get("ob_id") == "1650161"]
    assert len(ingested) == 1
    assert ingested[0]["entry_underlying"] == 78.95  # legacy value, not current 80.50


def test_ingest_places_new_stop_when_no_existing(tmp_path, monkeypatch, patched_state_file, patched_legacy_state):
    """Sanity check for the inverse path — placing a new stop when none exists."""
    trader = _make_trader(tmp_path)
    monkeypatch.setattr(mst, "DRY_RUN", False)
    monkeypatch.setattr(mst, "_find_existing_stop", lambda ob, u: None)

    set_stop_calls = []
    def _fake_set(pid):
        set_stop_calls.append(pid)
        trader.state["positions"][pid]["stop_order_id"] = "NEW_STOP_123"
    trader._set_stop_loss = _fake_set

    pos_id = trader.ingest_position(
        ob_id="2379768", units=97, entry_price=14.7,
        underlying_price=79.42, set_stop_loss=True,
    )

    assert pos_id is not None
    assert set_stop_calls == [pos_id]
    assert trader.state["positions"][pos_id]["stop_order_id"] == "NEW_STOP_123"
    assert trader.state["positions"][pos_id].get("stop_adopted") is not True


# ---------------------------------------------------------------------------
# _infer_direction
# ---------------------------------------------------------------------------


def test_infer_direction_bear_is_short():
    assert mst._infer_direction("BEAR SILVER X5 AVA 12") == "SHORT"


def test_infer_direction_mini_short_is_short():
    assert mst._infer_direction("MINI S SILVER AVA 401") == "SHORT"


def test_infer_direction_turbo_short_is_short():
    assert mst._infer_direction("TURBO S GOLD AVA 99") == "SHORT"


def test_infer_direction_default_long():
    assert mst._infer_direction("BULL SILVER X5 AVA 4") == "LONG"
    assert mst._infer_direction("MINI L SILVER AVA 331") == "LONG"
    assert mst._infer_direction("BULL GULD X8 N") == "LONG"
