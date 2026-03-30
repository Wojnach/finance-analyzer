"""Thread-safe TOTP authentication singleton for Avanza.

Wraps the ``avanza-api`` library's ``Avanza`` class with a double-checked
locking singleton so that the entire application shares one authenticated
session regardless of how many threads call ``get_instance()``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("portfolio.avanza.auth")


class AuthError(Exception):
    """Raised when Avanza authentication fails."""


def _create_avanza_client(credentials: dict[str, str]) -> Any:
    """Create and return an authenticated ``avanza.Avanza`` instance.

    Separated from :class:`AvanzaAuth` to allow easy mocking in tests
    (patch ``portfolio.avanza.auth._create_avanza_client``).

    Args:
        credentials: Dict with keys ``username``, ``password``, ``totpSecret``.

    Returns:
        An authenticated ``avanza.Avanza`` instance.

    Raises:
        AuthError: If authentication fails.
    """
    try:
        from avanza import Avanza  # noqa: WPS433 — late import

        client = Avanza(credentials, quiet=True)
        return client
    except Exception as exc:
        raise AuthError(f"Avanza authentication failed: {exc}") from exc


class AvanzaAuth:
    """Thread-safe singleton managing Avanza TOTP authentication.

    Usage::

        auth = AvanzaAuth.get_instance(username, password, totp_secret)
        auth.client  # -> avanza.Avanza instance

    Call ``AvanzaAuth.reset()`` to tear down the singleton (e.g. in tests or
    on session expiry).
    """

    _instance: AvanzaAuth | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        client: Any,
        push_subscription_id: str,
        csrf_token: str,
        authentication_session: str,
        customer_id: str,
    ) -> None:
        self.client = client
        self.push_subscription_id = push_subscription_id
        self.csrf_token = csrf_token
        self.authentication_session = authentication_session
        self.customer_id = customer_id

    @classmethod
    def get_instance(
        cls,
        username: str,
        password: str,
        totp_secret: str,
    ) -> AvanzaAuth:
        """Return the singleton, creating it on first call.

        Uses double-checked locking so that only the first caller pays the
        cost of TOTP authentication; subsequent callers return immediately.
        """
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            # Double-check after acquiring the lock
            if cls._instance is not None:
                return cls._instance

            credentials = {
                "username": username,
                "password": password,
                "totpSecret": totp_secret,
            }

            client = _create_avanza_client(credentials)

            instance = cls(
                client=client,
                push_subscription_id=getattr(client, "_push_subscription_id", ""),
                csrf_token=getattr(client, "_security_token", ""),
                authentication_session=getattr(client, "_authentication_session", ""),
                customer_id=getattr(client, "_customer_id", ""),
            )
            cls._instance = instance
            logger.info(
                "AvanzaAuth singleton created (customer_id=%s)",
                instance.customer_id,
            )
            return instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton — useful for tests or re-auth."""
        with cls._lock:
            cls._instance = None
            logger.info("AvanzaAuth singleton reset")
