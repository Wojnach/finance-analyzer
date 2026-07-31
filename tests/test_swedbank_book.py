"""Book load/valuation tests. Synthetic only — this repo is public."""

import pytest

from portfolio.swedbank.book import (
    Account,
    Book,
    BookError,
    Position,
    from_dict,
    revalue,
)
from portfolio.swedbank.pricing import PriceSweep, Quote


def _q(key, mark, currency="SEK", **kw):
    return Quote(
        key=key,
        mark=mark,
        currency=currency,
        source=kw.pop("source", "avanza"),
        mark_basis=kw.pop("mark_basis", "last"),
        **kw,
    )


def _sweep(quotes, fx=9.5, errors=None):
    s = PriceSweep()
    s.quotes = {q.key: q for q in quotes}
    s.fx = {"USDSEK": fx}
    s.errors = errors or {}
    return s


def _book(positions, cash=0.0, label="A"):
    return Book(accounts={label: Account(label=label, cash=cash, positions=positions)})


class TestPartialCostBasis:
    """Regression: P&L must not count value from positions that have no cost.

    Summing all market value against a partial cost basis inflates P&L by the
    entire value of every cost-less position — silently, with no error.
    """

    def test_costless_position_does_not_inflate_pnl(self):
        b = _book(
            [
                Position("INVE-B", 1, cost_basis=80.0, currency="SEK"),
                Position("SAAB-B", 1, cost_basis=None, currency="SEK"),
            ]
        )
        v = revalue(b, _sweep([_q("INVE-B", 100.0), _q("SAAB-B", 100.0)]))
        acc = v["accounts"]["A"]
        assert acc["holdings_value"] == pytest.approx(200.0)
        assert acc["cost_basis"] == pytest.approx(80.0)
        # +20 on the costed position, NOT +120.
        assert acc["pnl"] == pytest.approx(20.0)
        assert acc["pnl_pct"] == pytest.approx(25.0)
        assert acc["positions_without_cost_basis"] == 1
        assert acc["pnl_covers_value"] == pytest.approx(100.0)

    def test_all_costed_is_unaffected(self):
        b = _book(
            [
                Position("INVE-B", 1, cost_basis=80.0, currency="SEK"),
                Position("SAAB-B", 1, cost_basis=50.0, currency="SEK"),
            ]
        )
        acc = revalue(b, _sweep([_q("INVE-B", 100.0), _q("SAAB-B", 100.0)]))[
            "accounts"
        ]["A"]
        assert acc["pnl"] == pytest.approx(70.0)
        assert acc["positions_without_cost_basis"] == 0

    def test_no_cost_basis_at_all_reports_none(self):
        b = _book([Position("INVE-B", 1, cost_basis=None, currency="SEK")])
        acc = revalue(b, _sweep([_q("INVE-B", 100.0)]))["accounts"]["A"]
        assert acc["pnl"] is None
        assert acc["pnl_pct"] is None

    def test_consolidated_rollup_also_excludes_costless_value(self):
        b = Book(
            accounts={
                "A": Account(
                    "A", 0.0, [Position("INVE-B", 1, cost_basis=80.0, currency="SEK")]
                ),
                "B": Account(
                    "B", 0.0, [Position("INVE-B", 1, cost_basis=None, currency="SEK")]
                ),
            }
        )
        v = revalue(b, _sweep([_q("INVE-B", 100.0)]))
        row = next(r for r in v["consolidated"] if r["key"] == "INVE-B")
        assert row["qty"] == 2
        assert row["value"] == pytest.approx(200.0)
        assert row["pnl"] == pytest.approx(20.0)

    def test_grand_total_excludes_costless_value(self):
        b = Book(
            accounts={
                "A": Account(
                    "A", 0.0, [Position("INVE-B", 1, cost_basis=80.0, currency="SEK")]
                ),
                "B": Account(
                    "B", 0.0, [Position("SAAB-B", 1, cost_basis=None, currency="SEK")]
                ),
            }
        )
        t = revalue(b, _sweep([_q("INVE-B", 100.0), _q("SAAB-B", 100.0)]))["total"]
        assert t["holdings_value"] == pytest.approx(200.0)
        assert t["pnl"] == pytest.approx(20.0)
        assert t["positions_without_cost_basis"] == 1


class TestUnpriced:
    def test_unpriced_excluded_and_reported(self):
        b = _book(
            [
                Position("INVE-B", 1, cost_basis=80.0, currency="SEK"),
                Position("SAAB-B", 2, cost_basis=50.0, currency="SEK"),
            ]
        )
        v = revalue(b, _sweep([_q("INVE-B", 100.0)]))
        assert v["unpriced"] == [("A", "SAAB-B")]
        assert v["accounts"]["A"]["holdings_value"] == pytest.approx(100.0)

    def test_missing_fx_makes_usd_position_unpriced_not_wrong(self):
        b = _book([Position("NVDA", 1, cost_basis=100.0, currency="USD")])
        s = _sweep([_q("NVDA", 200.0, currency="USD")])
        s.fx = {}
        v = revalue(b, s)
        # Must NOT value it at 200 (fx defaulting to 1.0).
        assert v["unpriced"] == [("A", "NVDA")]
        assert v["accounts"]["A"]["holdings_value"] == 0.0

    def test_degraded_and_stale_surfaced(self):
        b = _book(
            [
                Position("INVE-B", 1, cost_basis=80.0, currency="SEK"),
                Position("SAAB-B", 1, cost_basis=80.0, currency="SEK"),
            ]
        )
        v = revalue(
            b,
            _sweep(
                [
                    _q("INVE-B", 100.0, degraded=True, source="alpaca:fallback"),
                    _q("SAAB-B", 100.0, stale_last=True, mark_basis="mid"),
                ]
            ),
        )
        assert v["degraded"] == [("A", "INVE-B", "alpaca:fallback")]
        assert v["stale_last"] == [("A", "SAAB-B")]


class TestFromDict:
    def test_rejects_unknown_instrument(self):
        with pytest.raises(BookError, match="unknown instrument"):
            from_dict(
                {
                    "schema": 1,
                    "accounts": {"A": {"positions": [{"key": "NOPE", "qty": 1}]}},
                }
            )

    def test_rejects_bad_schema(self):
        with pytest.raises(BookError, match="unsupported book schema"):
            from_dict({"schema": 99, "accounts": {}})

    @pytest.mark.parametrize("qty", [0, -3, 1.5, "2", None])
    def test_rejects_non_positive_int_qty(self, qty):
        with pytest.raises(BookError, match="positive int"):
            from_dict(
                {
                    "schema": 1,
                    "accounts": {"A": {"positions": [{"key": "NVDA", "qty": qty}]}},
                }
            )

    def test_currency_defaults_from_instrument_table(self):
        b = from_dict(
            {"schema": 1, "accounts": {"A": {"positions": [{"key": "NVDA", "qty": 1}]}}}
        )
        assert b.accounts["A"].positions[0].currency == "USD"
