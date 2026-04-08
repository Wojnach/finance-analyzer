"""Tests for cancel_stop_loss / cancel_all_stop_losses_for in avanza_session.

Covers the volume-constraint fix: stops must be cancelled (and verified
cleared) before a sell can be placed, otherwise Avanza rejects with
``short.sell.not.allowed``. The polling/verification step is the heart of
the fix — these tests pin its behavior on six scenarios.
"""

from unittest.mock import patch

import pytest

import portfolio.avanza_session as avs


# Helpers ------------------------------------------------------------------

def _sl(stop_id, ob_id, account_id="1625505"):
    """Build a fake Avanza stop-loss dict matching the live shape."""
    return {
        "id": stop_id,
        "status": "ACTIVE",
        "account": {"id": account_id},
        "orderbook": {"id": ob_id, "name": f"FAKE-{ob_id}"},
        "trigger": {"value": 1.0, "type": "LESS_OR_EQUAL"},
        "order": {"type": "SELL", "price": 0.95, "volume": 100},
        "deletable": True,
    }


# --- cancel_stop_loss --------------------------------------------------------


class TestCancelStopLossSingle:
    def test_empty_stop_id_returns_failed(self):
        result = avs.cancel_stop_loss("")
        assert result["status"] == "FAILED"
        assert result["http_status"] == 0

    def test_http_200_returns_success(self):
        with patch("portfolio.avanza_session.api_delete") as mock_del:
            mock_del.return_value = {"http_status": 200, "ok": True}
            result = avs.cancel_stop_loss("STOP_ABC", account_id="1625505")
        assert result["status"] == "SUCCESS"
        assert result["http_status"] == 200
        assert result["stop_id"] == "STOP_ABC"
        # Confirm the URL contains both account and stop id
        called_path = mock_del.call_args[0][0]
        assert "1625505" in called_path
        assert "STOP_ABC" in called_path

    def test_http_404_treated_as_success(self):
        """Already-gone stops are idempotent — 404 means goal already achieved."""
        with patch("portfolio.avanza_session.api_delete") as mock_del:
            mock_del.return_value = {"http_status": 404, "ok": True}
            result = avs.cancel_stop_loss("STOP_GONE")
        assert result["status"] == "SUCCESS"

    def test_http_500_returns_failed(self):
        with patch("portfolio.avanza_session.api_delete") as mock_del:
            mock_del.return_value = {"http_status": 500, "ok": False}
            result = avs.cancel_stop_loss("STOP_ERR")
        assert result["status"] == "FAILED"
        assert result["http_status"] == 500

    def test_exception_returns_failed_with_error_key(self):
        with patch("portfolio.avanza_session.api_delete", side_effect=RuntimeError("boom")):
            result = avs.cancel_stop_loss("STOP_X")
        assert result["status"] == "FAILED"
        assert result["http_status"] == 0
        assert "error" in result
        assert "boom" in result["error"]


# --- cancel_all_stop_losses_for ----------------------------------------------


class TestCancelAllStopLossesFor:
    def test_no_stops_present_returns_immediately(self):
        """Zero matching stops → SUCCESS without any DELETE calls."""
        with patch("portfolio.avanza_session.get_stop_losses_strict", return_value=[]):
            with patch("portfolio.avanza_session.api_delete") as mock_del:
                result = avs.cancel_all_stop_losses_for("9999999")
        assert result["status"] == "SUCCESS"
        assert result["cancelled"] == []
        assert result["remaining"] == []
        assert result["snapshot"] == []
        assert mock_del.call_count == 0

    def test_read_error_returns_failed_with_empty_snapshot(self):
        """If get_stop_losses_strict raises, FAIL CLOSED — do NOT silently
        report SUCCESS. Codex finding 1: an unread inventory is unknown,
        not empty."""
        with patch(
            "portfolio.avanza_session.get_stop_losses_strict",
            side_effect=RuntimeError("API down"),
        ):
            result = avs.cancel_all_stop_losses_for("OB1")
        assert result["status"] == "FAILED"
        assert result["cancelled"] == []
        assert result["snapshot"] == []
        assert "error" in result
        assert "read_error" in result["error"]

    def test_poll_read_error_returns_failed(self):
        """If the verification poll's read fails after we cancelled, treat
        as FAILED — we no longer know if the cancel actually took effect."""
        # Initial read succeeds, poll read raises
        side_effects = [
            [_sl("S1", "OB1")],  # initial
            RuntimeError("API down mid-poll"),  # poll
        ]
        def fake_strict():
            v = side_effects.pop(0)
            if isinstance(v, Exception):
                raise v
            return v
        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=fake_strict):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}):
                result = avs.cancel_all_stop_losses_for("OB1")
        assert result["status"] == "FAILED"
        # Snapshot is preserved so the caller can decide what to do
        assert len(result["snapshot"]) == 1
        # CRITICAL (codex finding): cancelled MUST be empty when the
        # verification poll failed. The DELETE was accepted by the broker
        # but never confirmed cleared, so it is unsafe to claim those
        # stops are gone.
        assert result["cancelled"] == []

    def test_single_stop_cleared_on_first_poll(self):
        """1 SL exists → cancel succeeds → re-query shows it cleared."""
        stops_state = [[_sl("S1", "OB1")], []]  # before, after
        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=stops_state):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}):
                result = avs.cancel_all_stop_losses_for("OB1")
        assert result["status"] == "SUCCESS"
        assert result["cancelled"] == ["S1"]
        assert result["remaining"] == []
        # Snapshot must be present so caller can re-arm if dependent sell fails
        assert len(result["snapshot"]) == 1
        assert result["snapshot"][0]["id"] == "S1"

    def test_multiple_stops_for_same_orderbook_all_cancelled(self):
        """3 cascade stops on same OB → all cancelled in one shot."""
        stops_state = [
            [_sl("S1", "OB1"), _sl("S2", "OB1"), _sl("S3", "OB1")],
            [],
        ]
        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=stops_state):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}) as mock_del:
                result = avs.cancel_all_stop_losses_for("OB1")
        assert result["status"] == "SUCCESS"
        assert sorted(result["cancelled"]) == ["S1", "S2", "S3"]
        assert mock_del.call_count == 3

    def test_filters_by_orderbook_id(self):
        """SLs across different orderbooks → only target ob_id touched."""
        stops_state = [
            [_sl("S1", "OB1"), _sl("S_other", "OB_OTHER"), _sl("S2", "OB1")],
            [_sl("S_other", "OB_OTHER")],
        ]
        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=stops_state):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}) as mock_del:
                result = avs.cancel_all_stop_losses_for("OB1")
        assert result["status"] == "SUCCESS"
        assert sorted(result["cancelled"]) == ["S1", "S2"]
        assert result["remaining"] == []  # OB_OTHER's stop is not "remaining" for OB1
        assert mock_del.call_count == 2  # OB_OTHER untouched

    def test_filters_by_account_id_when_specified(self):
        """SLs across multiple accounts → only target account is touched."""
        stops_state = [
            [_sl("S1", "OB1", "1625505"), _sl("S2", "OB1", "9999999")],
            [_sl("S2", "OB1", "9999999")],
        ]
        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=stops_state):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}) as mock_del:
                result = avs.cancel_all_stop_losses_for("OB1", account_id="1625505")
        assert result["status"] == "SUCCESS"
        assert result["cancelled"] == ["S1"]
        assert mock_del.call_count == 1

    def test_polls_until_clear(self):
        """Cancel returns 200 but propagation takes a few polls."""
        # 1 initial query, then 3 polls before it clears
        stops_state = [
            [_sl("S1", "OB1")],
            [_sl("S1", "OB1")],
            [_sl("S1", "OB1")],
            [],
        ]
        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=stops_state):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}):
                with patch("portfolio.avanza_session.time.sleep") as mock_sleep:
                    result = avs.cancel_all_stop_losses_for("OB1", max_wait=5.0, poll_interval=0.1)
        assert result["status"] == "SUCCESS"
        assert result["cancelled"] == ["S1"]
        assert result["remaining"] == []
        # Should have slept at least twice (between the 3 stale polls)
        assert mock_sleep.call_count >= 2

    def test_timeout_returns_partial(self):
        """Cancel succeeded but stop never disappears from list → PARTIAL."""
        # Always shows the stop, regardless of how many times we poll
        with patch("portfolio.avanza_session.get_stop_losses_strict", return_value=[_sl("S1", "OB1")]):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}):
                with patch("portfolio.avanza_session.time.sleep"):
                    with patch("portfolio.avanza_session.time.monotonic", side_effect=[0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]):
                        result = avs.cancel_all_stop_losses_for("OB1", max_wait=3.0, poll_interval=0.5)
        assert result["status"] == "PARTIAL"
        assert result["cancelled"] == ["S1"]
        assert result["remaining"] == ["S1"]

    def test_timeout_no_cancels_returns_failed(self):
        """Every cancel returned 500 AND the stop never cleared → FAILED."""
        with patch("portfolio.avanza_session.get_stop_losses_strict", return_value=[_sl("S1", "OB1")]):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 500, "ok": False}):
                with patch("portfolio.avanza_session.time.sleep"):
                    with patch("portfolio.avanza_session.time.monotonic", side_effect=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]):
                        result = avs.cancel_all_stop_losses_for("OB1", max_wait=2.0, poll_interval=0.5)
        assert result["status"] == "FAILED"
        assert result["cancelled"] == []
        assert result["remaining"] == ["S1"]

    def test_uses_per_stop_account_id_for_delete(self):
        """When SLs come from a different account, the cancel must use the
        SL's own account, not DEFAULT_ACCOUNT_ID. Otherwise the DELETE 404s
        because the account/stop combination doesn't match."""
        stops_state = [[_sl("S1", "OB1", account_id="9999999")], []]
        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=stops_state):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}) as mock_del:
                result = avs.cancel_all_stop_losses_for("OB1")  # no account filter
        assert result["status"] == "SUCCESS"
        called_path = mock_del.call_args[0][0]
        assert "9999999" in called_path
        assert "S1" in called_path

    def test_get_stop_losses_returns_non_dict_filtered(self):
        """Defensive: if Avanza returns garbage entries, skip them."""
        stops_state = [
            ["not a dict", None, _sl("S1", "OB1"), {"id": "no_orderbook"}],
            [],
        ]
        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=stops_state):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}) as mock_del:
                result = avs.cancel_all_stop_losses_for("OB1")
        assert result["status"] == "SUCCESS"
        assert result["cancelled"] == ["S1"]
        assert mock_del.call_count == 1

    def test_elapsed_seconds_present(self):
        with patch("portfolio.avanza_session.get_stop_losses_strict", return_value=[]):
            result = avs.cancel_all_stop_losses_for("OB1")
        assert "elapsed_seconds" in result
        assert isinstance(result["elapsed_seconds"], float)
        assert result["elapsed_seconds"] >= 0


# --- get_stop_losses_strict --------------------------------------------------


class TestGetStopLossesStrict:
    def test_returns_list_on_success(self):
        with patch("portfolio.avanza_session.api_get", return_value=[_sl("S1", "OB1")]):
            result = avs.get_stop_losses_strict()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_raises_on_api_error(self):
        with patch("portfolio.avanza_session.api_get", side_effect=RuntimeError("404")):
            with pytest.raises(RuntimeError):
                avs.get_stop_losses_strict()

    def test_raises_on_non_list_response(self):
        """Avanza occasionally returns {} or an error envelope on transient
        backend issues — must NOT silently treat that as 'no stops'."""
        with patch("portfolio.avanza_session.api_get", return_value={"error": "x"}):
            with pytest.raises(RuntimeError):
                avs.get_stop_losses_strict()


# --- rearm_stop_losses_from_snapshot -----------------------------------------


def _full_sl(stop_id="S1", ob_id="OB1", trigger=0.367, sell=0.350, volume=3566):
    """Build a complete stop-loss snapshot entry matching live Avanza shape."""
    return {
        "id": stop_id,
        "status": "ACTIVE",
        "account": {"id": "1625505"},
        "orderbook": {"id": ob_id},
        "trigger": {
            "value": trigger,
            "type": "LESS_OR_EQUAL",
            "validUntil": "2026-04-16",
            "valueType": "MONETARY",
        },
        "order": {
            "type": "SELL",
            "price": sell,
            "volume": volume,
            "validDays": 8,
            "priceType": "MONETARY",
        },
    }


class TestRearmStopLossesFromSnapshot:
    def test_empty_snapshot_returns_success(self):
        result = avs.rearm_stop_losses_from_snapshot([])
        assert result["status"] == "SUCCESS"
        assert result["rearmed"] == []
        assert result["failed"] == []

    def test_replaces_each_stop_with_new_id(self):
        snapshot = [_full_sl("OLD_S1", "OB1"), _full_sl("OLD_S2", "OB1", trigger=0.30)]
        place_calls = []

        def fake_place(orderbook_id, trigger_price, sell_price, volume, **kwargs):
            place_calls.append({
                "ob": orderbook_id,
                "trigger": trigger_price,
                "sell": sell_price,
                "vol": volume,
                **kwargs,
            })
            return {"status": "SUCCESS", "stoplossOrderId": f"NEW_{len(place_calls)}"}

        with patch("portfolio.avanza_session.place_stop_loss", side_effect=fake_place):
            result = avs.rearm_stop_losses_from_snapshot(snapshot)
        assert result["status"] == "SUCCESS"
        assert result["rearmed"] == ["NEW_1", "NEW_2"]
        assert result["failed"] == []
        assert len(place_calls) == 2
        # Verify the trigger/order params were preserved exactly
        assert place_calls[0]["trigger"] == 0.367
        assert place_calls[0]["sell"] == 0.350
        assert place_calls[0]["vol"] == 3566
        assert place_calls[0]["trigger_type"] == "LESS_OR_EQUAL"
        assert place_calls[0]["value_type"] == "MONETARY"

    def test_partial_failure_reports_partial(self):
        snapshot = [_full_sl("S1"), _full_sl("S2")]
        responses = [
            {"status": "SUCCESS", "stoplossOrderId": "NEW_1"},
            {"status": "FAILED"},
        ]
        with patch("portfolio.avanza_session.place_stop_loss", side_effect=responses):
            result = avs.rearm_stop_losses_from_snapshot(snapshot)
        assert result["status"] == "PARTIAL"
        assert result["rearmed"] == ["NEW_1"]
        assert result["failed"] == ["S2"]

    def test_total_failure_reports_failed(self):
        snapshot = [_full_sl("S1")]
        with patch("portfolio.avanza_session.place_stop_loss", side_effect=RuntimeError("nope")):
            result = avs.rearm_stop_losses_from_snapshot(snapshot)
        assert result["status"] == "FAILED"
        assert result["failed"] == ["S1"]
        assert result["rearmed"] == []

    def test_skips_entries_missing_required_fields(self):
        """Defensive: malformed snapshot entries are reported as failures
        rather than crashing the loop."""
        snapshot = [
            {"id": "BAD_NO_OB", "trigger": {"value": 1.0}, "order": {"price": 1.0, "volume": 100}},
            _full_sl("GOOD"),
            "not a dict",
            None,
        ]
        with patch("portfolio.avanza_session.place_stop_loss", return_value={"status": "SUCCESS", "stoplossOrderId": "NEW"}):
            result = avs.rearm_stop_losses_from_snapshot(snapshot)
        assert "GOOD" not in result["failed"]
        assert "BAD_NO_OB" in result["failed"]
        assert "NEW" in result["rearmed"]

    def test_uses_default_valid_days_when_validuntil_invalid(self):
        snapshot = [_full_sl("S1")]
        snapshot[0]["trigger"]["validUntil"] = "garbage"
        captured = {}
        def fake_place(**kwargs):
            captured.update(kwargs)
            return {"status": "SUCCESS", "stoplossOrderId": "X"}
        # The function uses positional args though, let me check
        def fake_place2(orderbook_id, trigger_price, sell_price, volume, **kwargs):
            captured.update(kwargs)
            captured["volume"] = volume
            return {"status": "SUCCESS", "stoplossOrderId": "X"}
        with patch("portfolio.avanza_session.place_stop_loss", side_effect=fake_place2):
            avs.rearm_stop_losses_from_snapshot(snapshot)
        # Default valid_days=8 should be used when validUntil cannot be parsed
        assert captured.get("valid_days") == 8


# --- import smoke test -------------------------------------------------------


def test_module_exports():
    """All new functions are importable from the module."""
    from portfolio.avanza_session import (
        cancel_all_stop_losses_for,
        cancel_stop_loss,
        get_stop_losses_strict,
        rearm_stop_losses_from_snapshot,
    )
    assert callable(cancel_stop_loss)
    assert callable(cancel_all_stop_losses_for)
    assert callable(get_stop_losses_strict)
    assert callable(rearm_stop_losses_from_snapshot)
