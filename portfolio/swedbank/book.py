"""Book persistence and valuation.

The book stores qty + cost basis + account cash. It NEVER stores market value —
value is always derived from a live sweep, so the file cannot go stale and be
mistaken for current.

`data/swedbank_book.json` holds real positions and is gitignored; this repo is
public. Nothing in this module may write position data anywhere else.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.swedbank.instruments import INSTRUMENTS
from portfolio.swedbank.pricing import value_holding

BOOK_PATH = "data/swedbank_book.json"
SCHEMA = 1
BASE = "SEK"


class BookError(ValueError):
    pass


@dataclass
class Position:
    key: str
    qty: int
    cost_basis: float | None = None
    currency: str = "SEK"

    def to_dict(self):
        return {
            "key": self.key,
            "qty": self.qty,
            "cost_basis_sek": self.cost_basis,
            "currency": self.currency,
        }


@dataclass
class Account:
    label: str
    cash: float = 0.0
    positions: list = field(default_factory=list)

    def to_dict(self):
        return {
            "cash": self.cash,
            "positions": [p.to_dict() for p in self.positions],
        }


@dataclass
class Book:
    accounts: dict = field(default_factory=dict)
    fx_at_snapshot: dict = field(default_factory=dict)
    snapshot_ts: str | None = None
    updated_at: str | None = None
    base_currency: str = BASE

    def to_dict(self):
        return {
            "schema": SCHEMA,
            "base_currency": self.base_currency,
            "snapshot_ts": self.snapshot_ts,
            "updated_at": self.updated_at,
            "fx_at_snapshot": dict(self.fx_at_snapshot),
            "accounts": {k: a.to_dict() for k, a in self.accounts.items()},
        }

    @property
    def keys_held(self):
        return sorted({p.key for a in self.accounts.values() for p in a.positions})


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def from_dict(raw):
    if not raw:
        raise BookError("empty book")
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise BookError(f"unsupported book schema {schema!r} (expected {SCHEMA})")
    accounts = {}
    for label, acc in (raw.get("accounts") or {}).items():
        positions = []
        for p in acc.get("positions") or []:
            key = p.get("key")
            if key not in INSTRUMENTS:
                raise BookError(
                    f"account {label!r} holds unknown instrument {key!r}; add it to "
                    f"portfolio/swedbank/instruments.py before loading this book"
                )
            qty = p.get("qty")
            if not isinstance(qty, int) or qty <= 0:
                raise BookError(
                    f"{label}/{key}: qty must be a positive int, got {qty!r}"
                )
            # cost_basis is ALWAYS in the book's base currency (SEK), never in
            # the instrument's quote currency. `currency` describes how the
            # instrument is QUOTED and is deliberately not applied to cost.
            # The book is hand-authored, so writing a USD cost basis for a USD
            # instrument is the natural mistake and would understate cost by the
            # FX rate — reporting roughly +900% P&L. Accept the explicit
            # cost_basis_sek spelling and sanity-check the ambiguous one.
            cost = p.get("cost_basis_sek")
            if cost is None:
                cost = p.get("cost_basis")
            if cost is not None:
                try:
                    cost = float(cost)
                except (TypeError, ValueError):
                    raise BookError(
                        f"{label}/{key}: cost_basis must be a number, got {cost!r}"
                    ) from None
                if cost <= 0:
                    raise BookError(
                        f"{label}/{key}: cost_basis must be positive, got {cost!r}"
                    )
            positions.append(
                Position(
                    key=key,
                    qty=qty,
                    cost_basis=cost,
                    currency=p.get("currency") or INSTRUMENTS[key].currency,
                )
            )
        accounts[label] = Account(
            label=label, cash=float(acc.get("cash") or 0.0), positions=positions
        )
    return Book(
        accounts=accounts,
        fx_at_snapshot=raw.get("fx_at_snapshot") or {},
        snapshot_ts=raw.get("snapshot_ts"),
        updated_at=raw.get("updated_at"),
        base_currency=raw.get("base_currency") or BASE,
    )


def load(path=BOOK_PATH):
    raw = load_json(path, default=None)
    if raw is None:
        raise BookError(
            f"no book at {path}. Create one with: "
            f"python -m portfolio.swedbank sync <export.md>"
        )
    return from_dict(raw)


def save(book, path=BOOK_PATH):
    book.updated_at = _now_iso()
    atomic_write_json(path, book.to_dict())
    return path


def revalue(book, sweep_result):
    """Value the book against a price sweep.

    Positions whose price is unavailable are reported in `unpriced` and excluded
    from totals rather than being valued at zero or at cost — a total that
    silently omits a position is less dangerous than one that invents a number,
    but only if the omission is visible.
    """
    fx = (sweep_result.fx or {}).get("USDSEK")
    out_accounts = {}
    unpriced, degraded, stale = [], [], []
    g_value = g_cost = g_cash = g_value_costed = 0.0
    g_costless = 0

    for label, acc in book.accounts.items():
        rows = []
        # a_value is every priced position. a_value_costed counts ONLY positions
        # that carry a cost basis, and is what P&L is computed against. Summing
        # all value against a partial cost would inflate P&L by the entire market
        # value of every cost-less position — e.g. two positions worth 100 each,
        # one with cost 80 and one with no cost, would report +120 instead of +20.
        a_value = a_cost = a_value_costed = 0.0
        n_costless = 0
        for pos in acc.positions:
            q = sweep_result.quotes.get(pos.key)
            if q is None:
                unpriced.append((label, pos.key))
                continue
            try:
                value = value_holding(pos.qty, q, fx, base=book.base_currency)
            except ValueError:
                unpriced.append((label, pos.key))
                continue
            if q.degraded:
                degraded.append((label, pos.key, q.source))
            if q.stale_last:
                stale.append((label, pos.key))
            pnl = value - pos.cost_basis if pos.cost_basis else None
            rows.append(
                {
                    "key": pos.key,
                    "name": INSTRUMENTS[pos.key].name,
                    "qty": pos.qty,
                    "mark": q.mark,
                    "mark_basis": q.mark_basis,
                    "currency": q.currency,
                    "spread_pct": q.spread_pct,
                    "age_s": q.age_s,
                    "source": q.source,
                    "degraded": q.degraded,
                    "stale_last": q.stale_last,
                    "value": value,
                    "cost_basis": pos.cost_basis,
                    "pnl": pnl,
                    "pnl_pct": (
                        (pnl / pos.cost_basis * 100.0)
                        if pnl is not None and pos.cost_basis
                        else None
                    ),
                    "avanza_ob": INSTRUMENTS[pos.key].avanza_ob,
                }
            )
            a_value += value
            if pos.cost_basis:
                a_cost += pos.cost_basis
                a_value_costed += value
            else:
                n_costless += 1
        rows.sort(key=lambda r: -r["value"])
        out_accounts[label] = {
            "cash": acc.cash,
            "holdings": rows,
            "holdings_value": a_value,
            "total_value": a_value + acc.cash,
            "cost_basis": a_cost,
            "pnl": a_value_costed - a_cost if a_cost else None,
            "pnl_pct": ((a_value_costed / a_cost - 1) * 100.0) if a_cost else None,
            # P&L covers only the costed subset. Surfaced so the UI can say so
            # rather than implying it covers the whole account.
            "pnl_covers_value": a_value_costed,
            "positions_without_cost_basis": n_costless,
        }
        g_value += a_value
        g_value_costed += a_value_costed
        g_cost += a_cost
        g_cash += acc.cash
        g_costless += n_costless

    return {
        "as_of": _now_iso(),
        "base_currency": book.base_currency,
        "fx": sweep_result.fx,
        "accounts": out_accounts,
        "total": {
            "holdings_value": g_value,
            "cash": g_cash,
            "total_value": g_value + g_cash,
            "cost_basis": g_cost,
            "pnl": g_value_costed - g_cost if g_cost else None,
            "pnl_pct": ((g_value_costed / g_cost - 1) * 100.0) if g_cost else None,
            "pnl_covers_value": g_value_costed,
            "positions_without_cost_basis": g_costless,
        },
        "consolidated": _consolidate(out_accounts),
        "unpriced": unpriced,
        "degraded": degraded,
        "stale_last": stale,
        "price_errors": dict(sweep_result.errors),
        "sweep_duration_s": sweep_result.duration_s,
    }


def _consolidate(out_accounts):
    """Same instrument held across accounts, rolled up."""
    agg = {}
    for acc in out_accounts.values():
        for r in acc["holdings"]:
            e = agg.setdefault(
                r["key"],
                {
                    "key": r["key"],
                    "name": r["name"],
                    "qty": 0,
                    "value": 0.0,
                    "cost_basis": 0.0,
                    "_value_costed": 0.0,
                    "mark": r["mark"],
                    "currency": r["currency"],
                    "avanza_ob": r["avanza_ob"],
                },
            )
            e["qty"] += r["qty"]
            e["value"] += r["value"]
            if r["cost_basis"]:
                e["cost_basis"] += r["cost_basis"]
                e["_value_costed"] += r["value"]
    for e in agg.values():
        vc = e.pop("_value_costed")
        e["pnl"] = vc - e["cost_basis"] if e["cost_basis"] else None
        e["pnl_pct"] = (vc / e["cost_basis"] - 1) * 100.0 if e["cost_basis"] else None
    return sorted(agg.values(), key=lambda e: -e["value"])
