"""Tests for portfolio.avanza.auth — thread-safe TOTP singleton."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AuthError, AvanzaAuth, _create_avanza_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure every test starts with a clean singleton."""
    AvanzaAuth.reset()
    yield
    AvanzaAuth.reset()


def _make_mock_client():
    """Create a mock that mimics an authenticated avanza.Avanza instance."""
    client = MagicMock()
    client._push_subscription_id = "push-sub-123"
    client._security_token = "csrf-token-abc"
    client._authentication_session = "auth-sess-xyz"
    client._customer_id = "cust-42"
    return client


# ---------------------------------------------------------------------------
# Singleton creation
# ---------------------------------------------------------------------------

class TestSingletonCreation:
    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_creates_instance(self, mock_create):
        mock_create.return_value = _make_mock_client()

        auth = AvanzaAuth.get_instance("user", "pass", "secret")

        assert auth is not None
        assert auth.customer_id == "cust-42"
        assert auth.push_subscription_id == "push-sub-123"
        assert auth.csrf_token == "csrf-token-abc"
        assert auth.authentication_session == "auth-sess-xyz"
        mock_create.assert_called_once_with({
            "username": "user",
            "password": "pass",
            "totpSecret": "secret",
        })

    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_returns_same_instance(self, mock_create):
        mock_create.return_value = _make_mock_client()

        a1 = AvanzaAuth.get_instance("user", "pass", "secret")
        a2 = AvanzaAuth.get_instance("user", "pass", "secret")

        assert a1 is a2
        # Should only authenticate once
        assert mock_create.call_count == 1

    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_client_attribute(self, mock_create):
        mock_client = _make_mock_client()
        mock_create.return_value = mock_client

        auth = AvanzaAuth.get_instance("user", "pass", "secret")
        assert auth.client is mock_client


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_ten_threads_get_same_instance(self, mock_create):
        mock_create.return_value = _make_mock_client()

        results: list[AvanzaAuth] = []
        barrier = threading.Barrier(10)

        def get_auth():
            barrier.wait()  # All threads start together
            auth = AvanzaAuth.get_instance("user", "pass", "secret")
            results.append(auth)

        threads = [threading.Thread(target=get_auth) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 10
        # All references should be the same object
        assert all(r is results[0] for r in results)
        # Factory should have been called exactly once
        assert mock_create.call_count == 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_reset_clears_singleton(self, mock_create):
        mock_create.return_value = _make_mock_client()

        a1 = AvanzaAuth.get_instance("user", "pass", "secret")
        AvanzaAuth.reset()

        mock_create.return_value = _make_mock_client()
        a2 = AvanzaAuth.get_instance("user", "pass", "secret")

        assert a1 is not a2
        assert mock_create.call_count == 2

    def test_reset_on_empty_singleton(self):
        # Should not raise
        AvanzaAuth.reset()


# ---------------------------------------------------------------------------
# Auth errors
# ---------------------------------------------------------------------------

class TestAuthError:
    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_auth_error_on_bad_creds(self, mock_create):
        mock_create.side_effect = AuthError("bad credentials")

        with pytest.raises(AuthError, match="bad credentials"):
            AvanzaAuth.get_instance("bad", "creds", "nope")

        # Singleton should NOT be set after failure
        assert AvanzaAuth._instance is None

    def test_create_avanza_client_wraps_exception(self):
        """_create_avanza_client wraps library exceptions in AuthError."""
        with patch("portfolio.avanza.auth.Avanza", create=True) as MockAvanza:
            # Patch at the import target inside the function
            pass

        # More direct: patch the import
        with patch.dict("sys.modules", {"avanza": MagicMock()}):
            import sys
            mock_mod = sys.modules["avanza"]
            mock_mod.Avanza.side_effect = RuntimeError("connection refused")

            with pytest.raises(AuthError, match="connection refused"):
                _create_avanza_client({"username": "u", "password": "p", "totpSecret": "s"})
