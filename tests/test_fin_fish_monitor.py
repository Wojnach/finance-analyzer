"""Tests for scripts/fin_fish_monitor.py — focused on AV-P1-1 URL bug.

The fish monitor calls `api_delete` to cancel a stop-loss when adjusting
volume on partial fill. The DELETE URL must include both `accountId` and
`stopId` per the canonical pattern:

    /_api/trading/stoploss/{accountId}/{stopId}

Same pattern is used in:
- portfolio.avanza_session.cancel_stop_loss (line 911)
- portfolio.avanza_control.delete_stop_loss (line 227)
- data.metals_avanza_helpers.delete_stop_loss (line 472)

Prior to the AV-P1-1 fix, this script used `/_api/trading/stoploss/{stop_id}`
(missing accountId), which returned 404. Avanza-side, the stop wasn't
cancelled — the script logged a warning and continued, leaving the partial-fill
adjustment incomplete with two SLs (old + new) racing on the position.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def monitor_module():
    """Import scripts/fin_fish_monitor.py with avanza_session pre-mocked."""
    # Pre-mock avanza_session so import doesn't try to hit the live API.
    fake_session = MagicMock()
    fake_session.api_get = MagicMock()
    fake_session.api_post = MagicMock()
    fake_session.api_delete = MagicMock()
    fake_session.get_instrument_price = MagicMock()
    fake_session.get_positions = MagicMock()
    fake_session.verify_session = MagicMock()
    fake_session._get_csrf = MagicMock()
    fake_session.place_sell_order = MagicMock()
    sys.modules["portfolio.avanza_session"] = fake_session

    # Now import the script.
    if "fin_fish_monitor" in sys.modules:
        importlib.reload(sys.modules["fin_fish_monitor"])
    import fin_fish_monitor as mod
    return mod


class TestDeleteStopLossURL:
    """AV-P1-1 regression: the URL must be the canonical 2-segment shape."""

    def test_url_includes_account_id_and_stop_id(self, monitor_module):
        """The api_delete call must use /_api/trading/stoploss/{account}/{stop_id}.

        Before the fix, the path was /_api/trading/stoploss/{stop_id} which
        returned 404 silently.
        """
        with patch("portfolio.avanza_session.api_delete") as mock_delete:
            mock_delete.return_value = {"ok": True}
            monitor_module.delete_stop_loss("ACC-1625505", "SL-ABC123")

        mock_delete.assert_called_once()
        called_path = mock_delete.call_args[0][0]
        # The canonical shape per cancel_stop_loss in avanza_session.py:911 is
        # /_api/trading/stoploss/{accountId}/{stopId}.
        assert "/_api/trading/stoploss/" in called_path
        assert "ACC-1625505" in called_path, (
            f"accountId missing from DELETE URL: {called_path!r} — "
            "fin_fish_monitor used the buggy 1-segment URL"
        )
        assert "SL-ABC123" in called_path
        # Order matters: /stoploss/{account}/{stop}
        assert called_path.endswith("/ACC-1625505/SL-ABC123") or \
               called_path == f"/_api/trading/stoploss/ACC-1625505/SL-ABC123", (
                   f"DELETE path must be /_api/trading/stoploss/<account>/<stop>, "
                   f"got {called_path!r}"
               )

    def test_returns_true_on_ok_response(self, monitor_module):
        with patch("portfolio.avanza_session.api_delete") as mock_delete:
            mock_delete.return_value = {"ok": True}
            ok = monitor_module.delete_stop_loss("ACC-1625505", "SL-1")
        assert ok is True

    def test_returns_false_on_failure(self, monitor_module):
        with patch("portfolio.avanza_session.api_delete") as mock_delete:
            mock_delete.return_value = {"ok": False, "http_status": 500}
            ok = monitor_module.delete_stop_loss("ACC-1625505", "SL-1")
        assert ok is False

    def test_returns_false_on_exception(self, monitor_module):
        with patch("portfolio.avanza_session.api_delete",
                   side_effect=RuntimeError("network down")):
            ok = monitor_module.delete_stop_loss("ACC-1625505", "SL-1")
        assert ok is False
