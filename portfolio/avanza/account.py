"""Account data — positions, buying power, transactions.

Typed wrappers around :class:`~portfolio.avanza.client.AvanzaClient`
raw delegators for account-level queries.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Sequence

from avanza.constants import TransactionsDetailsType

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import AccountCash, Position, Transaction

logger = logging.getLogger("portfolio.avanza.account")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_positions(account_id: str | None = None) -> list[Position]:
    """Fetch all positions, optionally filtered to a single account.

    Args:
        account_id: When provided only positions for this account are
            returned.  Otherwise all positions across all accounts.

    Returns:
        List of :class:`~portfolio.avanza.types.Position`.
    """
    client = AvanzaClient.get_instance()
    raw: Any = client.get_positions_raw()

    # The API may return a dict with a nested positions list, or a list.
    positions_raw: list[dict[str, Any]]
    if isinstance(raw, dict):
        # Prefer "withOrderbook" (newer API), fall back to "positions"
        positions_raw = raw.get("withOrderbook", raw.get("positions", []))
    elif isinstance(raw, list):
        positions_raw = raw
    else:
        positions_raw = []

    positions = [Position.from_api(p) for p in positions_raw]

    if account_id is not None:
        positions = [p for p in positions if p.account_id == str(account_id)]

    logger.debug(
        "get_positions account_id=%s total=%d filtered=%d",
        account_id,
        len(positions_raw),
        len(positions),
    )
    return positions


def get_buying_power(account_id: str | None = None) -> AccountCash:
    """Fetch buying power / cash info for a specific account.

    Args:
        account_id: Account to query.  Defaults to the client's configured
            account.

    Returns:
        :class:`~portfolio.avanza.types.AccountCash`.
    """
    client = AvanzaClient.get_instance()
    acct = str(account_id) if account_id else client.account_id
    raw: Any = client.get_overview_raw()

    # The overview contains a list of accounts — find the right one.
    accounts: list[dict[str, Any]]
    if isinstance(raw, dict):
        accounts = raw.get("accounts", [])
    elif isinstance(raw, list):
        accounts = raw
    else:
        accounts = []

    for account in accounts:
        if str(account.get("accountId", account.get("id", ""))) == acct:
            logger.debug("get_buying_power account_id=%s found", acct)
            return AccountCash.from_api(account)

    # Account not found — return zeroes
    logger.warning("get_buying_power account_id=%s not found in overview", acct)
    return AccountCash(buying_power=0.0, total_value=0.0, own_capital=0.0)


def get_transactions(
    from_date: str,
    to_date: str,
    types: Sequence[str] | None = None,
    account_id: str | None = None,
) -> list[Transaction]:
    """Fetch historical transactions.

    Args:
        from_date: Start date (ISO-8601, e.g. ``"2026-01-01"``).
        to_date: End date (ISO-8601).
        types: Transaction type filters (e.g. ``["BUY", "SELL"]``).
            When *None* all types are returned.
        account_id: Unused by the library call but kept for future
            server-side filtering.

    Returns:
        List of :class:`~portfolio.avanza.types.Transaction`.
    """
    client = AvanzaClient.get_instance()

    tx_types: list[TransactionsDetailsType] | None = None
    if types:
        tx_types = [TransactionsDetailsType(t) for t in types]

    raw: Any = client.avanza.get_transactions_details(
        transaction_details_types=tx_types or [],
        transactions_from=date.fromisoformat(from_date),
        transactions_to=date.fromisoformat(to_date),
    )

    # The API may return a dict with a "transactions" key, or a list.
    tx_list: list[dict[str, Any]]
    if isinstance(raw, dict):
        tx_list = raw.get("transactions", [])
    elif isinstance(raw, list):
        tx_list = raw
    else:
        tx_list = []

    transactions = [Transaction.from_api(t) for t in tx_list]

    if account_id is not None:
        transactions = [t for t in transactions if t.account_id == str(account_id)]

    logger.debug(
        "get_transactions from=%s to=%s types=%s count=%d",
        from_date,
        to_date,
        types,
        len(transactions),
    )
    return transactions
