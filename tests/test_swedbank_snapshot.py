"""Swedbank snapshot derivation tests.

All fixtures are SYNTHETIC. This repository is public — never seed a test from
the live book (`data/swedbank_book.json`), and never paste real quantities,
cost basis or account totals into this file.
"""

import pytest

from portfolio.swedbank.snapshot import (
    Holding,
    RawRow,
    SnapshotError,
    derive_holdings,
    diff_holdings,
    parse_markdown_table,
    parse_number,
    reconcile,
    solve_fx,
)

SYNTH_FX = 8.5


def _row(name, price, qty, currency="USD", pct=None):
    """Build a row the way a broker export would present it."""
    rate = SYNTH_FX if currency != "SEK" else 1.0
    return RawRow(
        name=name,
        price=price,
        currency=currency,
        value_local=round(price * qty * rate, 2),
        since_purch_pct=pct,
    )


class TestParseNumber:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1 015,89", 1015.89),
            ("1 015,89", 1015.89),
            ("1 015,89", 1015.89),
            ("338,19", 338.19),
            ("−0,56", -0.56),
            ("–12,53", -12.53),
            ("+30,86", 30.86),
            ("-37,76%", -37.76),
            ("13,20", 13.20),
            ("591,50", 591.50),
            ("0", 0.0),
        ],
    )
    def test_broker_formats(self, raw, expected):
        assert parse_number(raw) == pytest.approx(expected)

    def test_unicode_minus_is_not_silently_dropped(self):
        # U+2212 is what the broker actually emits. If it were stripped rather
        # than converted, a loss would read as a gain.
        assert parse_number("−42,87") == pytest.approx(-42.87)

    @pytest.mark.parametrize("bad", ["", "   ", "—", "abc", None])
    def test_rejects_garbage(self, bad):
        with pytest.raises(SnapshotError):
            parse_number(bad)


class TestSolveFx:
    def test_recovers_known_rate(self):
        rows = [
            _row("Alpha", 338.19, 1),
            _row("Beta", 190.01, 32),
            _row("Gamma", 739.00, 2),
            _row("Delta", 1015.89, 1),
            _row("Epsilon", 23.22, 4),
        ]
        fx, diag = solve_fx(rows, reference_fx=SYNTH_FX)
        assert fx == pytest.approx(SYNTH_FX, rel=1e-6)
        assert diag["integer_hits"] == 5
        assert diag["unanimous"]

    def test_local_only_book_needs_no_fx(self):
        rows = [_row("Local", 100.0, 7, currency="SEK")]
        fx, diag = solve_fx(rows, reference_fx=SYNTH_FX)
        assert fx == 1.0
        assert diag["foreign_rows"] == 0

    def test_mixed_currency_ignores_local_rows(self):
        rows = [
            _row("Local", 408.50, 34, currency="SEK"),
            _row("Foreign", 190.01, 32),
            _row("Foreign2", 739.00, 6),
        ]
        fx, _ = solve_fx(rows, reference_fx=SYNTH_FX)
        assert fx == pytest.approx(SYNTH_FX, rel=1e-6)

    def test_refuses_to_guess_when_no_rate_explains_all_rows(self):
        # One row perturbed so no single rate makes everything integer. The
        # solver must fail loudly: a wrong rate yields wrong quantities with
        # no other symptom.
        rows = [
            _row("Alpha", 190.01, 32),
            _row("Beta", 739.00, 2),
            RawRow("Corrupt", 100.0, "USD", 1234.567, None),
        ]
        with pytest.raises(SnapshotError, match="no single FX rate"):
            solve_fx(rows, reference_fx=SYNTH_FX)

    def test_rejects_nonpositive_anchor(self):
        with pytest.raises(SnapshotError):
            solve_fx([RawRow("Bad", 0.0, "USD", 100.0, None)], reference_fx=SYNTH_FX)

    def test_fractional_share_count_is_rejected(self):
        rows = [
            RawRow("Half", 100.0, "USD", 100.0 * 2.5 * SYNTH_FX, None),
            RawRow("Whole", 50.0, "USD", 50.0 * 3 * SYNTH_FX, None),
        ]
        # 2.5 shares cannot be reconciled with an integer-share model at the
        # same rate that makes the other row whole.
        with pytest.raises(SnapshotError):
            solve_fx(rows, reference_fx=SYNTH_FX)


class TestDeriveHoldings:
    def test_integer_quantities_and_cost_basis(self):
        rows = [
            _row("Alpha", 200.0, 5, pct=25.0),
            _row("Local", 100.0, 10, currency="SEK", pct=-50.0),
        ]
        fx, _ = solve_fx(rows, reference_fx=SYNTH_FX)
        holdings = derive_holdings(rows, fx, key_for=lambda n: n.upper())
        by = {h.key: h for h in holdings}
        assert by["ALPHA"].qty == 5
        assert by["LOCAL"].qty == 10
        # 200*5*8.5 = 8500 at +25% -> cost 6800
        assert by["ALPHA"].cost_basis_local == pytest.approx(6800.0)
        # 100*10 = 1000 at -50% -> cost 2000
        assert by["LOCAL"].cost_basis_local == pytest.approx(2000.0)

    def test_missing_pct_leaves_cost_none(self):
        rows = [_row("NoPct", 100.0, 3)]
        fx, _ = solve_fx(rows, reference_fx=SYNTH_FX)
        h = derive_holdings(rows, fx, key_for=str)[0]
        assert h.cost_basis_local is None

    def test_total_loss_percentage_rejected(self):
        rows = [_row("Wiped", 100.0, 1, pct=-100.0)]
        fx, _ = solve_fx(rows, reference_fx=SYNTH_FX)
        with pytest.raises(SnapshotError, match="non-positive cost basis"):
            derive_holdings(rows, fx, key_for=str)

    def test_wrong_fx_produces_error_not_silent_rounding(self):
        rows = [_row("Alpha", 333.33, 7)]
        with pytest.raises(SnapshotError, match="integer share count"):
            derive_holdings(rows, 1.234, key_for=str)


class TestReconcile:
    def _holdings(self):
        rows = [_row("A", 100.0, 4, pct=100.0), _row("B", 50.0, 2, pct=0.0)]
        fx, _ = solve_fx(rows, reference_fx=SYNTH_FX)
        return derive_holdings(rows, fx, key_for=str)

    def test_value_matches_stated(self):
        h = self._holdings()
        total = sum(x.value_at_snapshot for x in h)
        rep = reconcile(h, stated_value=total, cash=0.0)
        assert rep["value_ok"]
        assert rep["value_delta"] == pytest.approx(0.0)

    def test_cash_included_in_value(self):
        h = self._holdings()
        total = sum(x.value_at_snapshot for x in h)
        rep = reconcile(h, stated_value=total + 1234.5, cash=1234.5)
        assert rep["value_ok"]

    def test_detects_mismatch(self):
        h = self._holdings()
        rep = reconcile(h, stated_value=1.0)
        assert rep["value_ok"] is False


class TestDiff:
    def _h(self, key, qty):
        return Holding(key, key, qty, "USD", 100.0, 10.0, 10.0 * qty)

    def test_detects_open_close_change(self):
        old = [self._h("A", 5), self._h("B", 3), self._h("C", 1)]
        new = [self._h("A", 8), self._h("C", 1), self._h("D", 2)]
        d = diff_holdings(old, new)
        assert [x.key for x in d["opened"]] == ["D"]
        assert [x.key for x in d["closed"]] == ["B"]
        assert [(o.qty, n.qty) for o, n in d["changed"]] == [(5, 8)]
        assert [x.key for x in d["unchanged"]] == ["C"]

    def test_idempotent_on_identical_input(self):
        book = [self._h("A", 5), self._h("B", 3)]
        d = diff_holdings(book, book)
        assert not d["opened"] and not d["closed"] and not d["changed"]
        assert len(d["unchanged"]) == 2


class TestParseMarkdownTable:
    TABLE = """
| Holding | Last | Today % | Since purch. % | Value SEK |
|---|---|---|---|---|
| Alpha Corp | 338,19 USD | −0,56 | +30,86 | 3 263,65 |
| Local AB | 13,20 | +0,76 | −12,53 | 9 240,00 |
| **Equities** | | | | **175 415,77** |
| Tracker One | 146,92 | +0,83 | +3,60 | 36 730,00 |
"""

    def test_extracts_rows_and_skips_subtotals(self):
        rows = parse_markdown_table(self.TABLE)
        assert [r.name for r in rows] == ["Alpha Corp", "Local AB", "Tracker One"]

    def test_currency_detected_from_price_cell(self):
        rows = parse_markdown_table(self.TABLE)
        assert rows[0].currency == "USD"
        assert rows[1].currency == "SEK"

    def test_negative_since_purchase_parsed(self):
        rows = parse_markdown_table(self.TABLE)
        assert rows[1].since_purch_pct == pytest.approx(-12.53)

    def test_header_and_separator_ignored(self):
        rows = parse_markdown_table(self.TABLE)
        assert all(r.name.lower() != "holding" for r in rows)
