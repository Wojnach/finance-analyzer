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
        with patch("portfolio.avanza_session.get_stop_losses", return_value=[]):
            with patch("portfolio.avanza_session.api_delete") as mock_del:
                result = avs.cancel_all_stop_losses_for("9999999")
        assert result["status"] == "SUCCESS"
        assert result["cancelled"] == []
        assert result["remaining"] == []
        assert mock_del.call_count == 0

    def test_single_stop_cleared_on_first_poll(self):
        """1 SL exists → cancel succeeds → re-query shows it cleared."""
        stops_state = [[_sl("S1", "OB1")], []]  # before, after
        with patch("portfolio.avanza_session.get_stop_losses", side_effect=stops_state):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}):
                result = avs.cancel_all_stop_losses_for("OB1")
        assert result["status"] == "SUCCESS"
        assert result["cancelled"] == ["S1"]
        assert result["remaining"] == []

    def test_multiple_stops_for_same_orderbook_all_cancelled(self):
        """3 cascade stops on same OB → all cancelled in one shot."""
        stops_state = [
            [_sl("S1", "OB1"), _sl("S2", "OB1"), _sl("S3", "OB1")],
            [],
        ]
        with patch("portfolio.avanza_session.get_stop_losses", side_effect=stops_state):
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
        with patch("portfolio.avanza_session.get_stop_losses", side_effect=stops_state):
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
        with patch("portfolio.avanza_session.get_stop_losses", side_effect=stops_state):
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
        with patch("portfolio.avanza_session.get_stop_losses", side_effect=stops_state):
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
        with patch("portfolio.avanza_session.get_stop_losses", return_value=[_sl("S1", "OB1")]):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}):
                with patch("portfolio.avanza_session.time.sleep"):
                    with patch("portfolio.avanza_session.time.monotonic", side_effect=[0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]):
                        result = avs.cancel_all_stop_losses_for("OB1", max_wait=3.0, poll_interval=0.5)
        assert result["status"] == "PARTIAL"
        assert result["cancelled"] == ["S1"]
        assert result["remaining"] == ["S1"]

    def test_timeout_no_cancels_returns_failed(self):
        """Every cancel returned 500 AND the stop never cleared → FAILED."""
        with patch("portfolio.avanza_session.get_stop_losses", return_value=[_sl("S1", "OB1")]):
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
        with patch("portfolio.avanza_session.get_stop_losses", side_effect=stops_state):
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
        with patch("portfolio.avanza_session.get_stop_losses", side_effect=stops_state):
            with patch("portfolio.avanza_session.api_delete", return_value={"http_status": 200, "ok": True}) as mock_del:
                result = avs.cancel_all_stop_losses_for("OB1")
        assert result["status"] == "SUCCESS"
        assert result["cancelled"] == ["S1"]
        assert mock_del.call_count == 1

    def test_elapsed_seconds_present(self):
        with patch("portfolio.avanza_session.get_stop_losses", return_value=[]):
            result = avs.cancel_all_stop_losses_for("OB1")
        assert "elapsed_seconds" in result
        assert isinstance(result["elapsed_seconds"], float)
        assert result["elapsed_seconds"] >= 0


# --- import smoke test -------------------------------------------------------


def test_module_exports():
    """Both new functions are importable from the module."""
    from portfolio.avanza_session import cancel_all_stop_losses_for, cancel_stop_loss
    assert callable(cancel_stop_loss)
    assert callable(cancel_all_stop_losses_for)
