"""Swedbank book — monitoring-only ledger for externally-custodied holdings.

This subsystem tracks share accounts held at an external broker. It re-prices
them, computes signals and trajectories, and feeds a dashboard tab.

It NEVER places, modifies or cancels an order. The operator executes manually.
`tests/test_swedbank_no_trading.py` asserts that no order-placing symbol is
reachable from this package.

Privacy: the live book (`data/swedbank_book.json`) holds real positions and is
gitignored. This repository is public. Never commit real quantities, cost basis,
account totals or account labels, and never seed a test fixture from the live
book.
"""

from portfolio.swedbank.instruments import (
    INSTRUMENTS,
    AssetClass,
    Instrument,
    by_key,
)

__all__ = ["INSTRUMENTS", "AssetClass", "Instrument", "by_key"]
