"""Singleton HTTP client wrapping the avanza-api library.

Provides raw delegator methods that return whatever the underlying
``avanza.Avanza`` instance returns.  Typed higher-level modules (market
data, trading, account, etc.) will wrap these delegators and return our
own dataclasses from :mod:`portfolio.avanza.types`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from portfolio.avanza.auth import AvanzaAuth

logger = logging.getLogger("portfolio.avanza.client")

DEFAULT_ACCOUNT_ID = "1625505"


class AvanzaClient:
    """Singleton client wrapping the avanza-api library.

    Usage::

        client = AvanzaClient.get_instance(config)
        raw = client.get_market_data_raw("2213050")
    """

    _instance: AvanzaClient | None = None
    _lock = threading.Lock()

    def __init__(self, auth: AvanzaAuth, account_id: str) -> None:
        self._auth = auth
        self._account_id = account_id

    @classmethod
    def get_instance(cls, config: dict[str, Any] | None = None) -> AvanzaClient:
        """Return the singleton, creating it on first call.

        Args:
            config: Application config dict.  Must contain an ``"avanza"`` key
                with ``"username"``, ``"password"``, and ``"totp_secret"`` when
                creating for the first time.  Ignored on subsequent calls.
        """
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            if config is None:
                raise ValueError(
                    "AvanzaClient.get_instance() requires config on first call"
                )

            avanza_cfg = config.get("avanza", {})
            auth = AvanzaAuth.get_instance(
                username=avanza_cfg["username"],
                password=avanza_cfg["password"],
                totp_secret=avanza_cfg["totp_secret"],
            )
            account_id = str(avanza_cfg.get("account_id", DEFAULT_ACCOUNT_ID))

            instance = cls(auth=auth, account_id=account_id)
            cls._instance = instance
            logger.info(
                "AvanzaClient singleton created (account_id=%s)", account_id
            )
            return instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton."""
        with cls._lock:
            cls._instance = None
            logger.info("AvanzaClient singleton reset")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def avanza(self) -> Any:
        """The underlying ``avanza.Avanza`` instance."""
        return self._auth.client

    @property
    def push_subscription_id(self) -> str:
        return self._auth.push_subscription_id

    @property
    def csrf_token(self) -> str:
        return self._auth.csrf_token

    @property
    def session(self) -> Any:
        """The underlying ``requests.Session`` used by the avanza-api library."""
        return self._auth.client._session

    # ------------------------------------------------------------------
    # Raw delegators — return whatever the library returns
    # ------------------------------------------------------------------

    def get_positions_raw(self) -> Any:
        return self.avanza.get_accounts_positions()

    def get_overview_raw(self) -> Any:
        return self.avanza.get_overview()

    def get_market_data_raw(self, ob_id: str) -> Any:
        return self.avanza.get_market_data(ob_id)

    def get_order_book_raw(self, ob_id: str) -> Any:
        return self.avanza.get_order_book(ob_id)

    def get_deals_raw(self) -> Any:
        return self.avanza.get_deals()

    def get_orders_raw(self) -> Any:
        return self.avanza.get_orders()

    def get_all_stop_losses_raw(self) -> Any:
        return self.avanza.get_all_stop_losses()

    def get_news_raw(self, ob_id: str) -> Any:
        return self.avanza.get_news(ob_id)
