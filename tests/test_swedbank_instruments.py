"""Guards on the pinned instrument table.

The offline tests here enforce the invariants the premortem identified (P0-2,
P1-4, P1-5). The single network test verifies each pinned orderbook ID still
resolves to the instrument we think it is; it is marked `integration` so the
normal suite stays offline.
"""

import pytest

from portfolio.swedbank.instruments import (
    INSTRUMENTS,
    AssetClass,
    by_key,
    by_orderbook,
)


class TestTableIntegrity:
    def test_keys_match_their_entries(self):
        for key, inst in INSTRUMENTS.items():
            assert inst.key == key

    def test_orderbook_ids_are_unique(self):
        # A duplicate ID would silently price two positions from one instrument.
        seen = {}
        for inst in INSTRUMENTS.values():
            assert inst.avanza_ob not in seen, (
                f"{inst.key} and {seen.get(inst.avanza_ob)} share orderbook "
                f"{inst.avanza_ob}"
            )
            seen[inst.avanza_ob] = inst.key

    def test_orderbook_ids_are_numeric_strings(self):
        for inst in INSTRUMENTS.values():
            assert inst.avanza_ob.isdigit(), f"{inst.key}: {inst.avanza_ob!r}"

    def test_every_instrument_has_an_asset_class(self):
        for inst in INSTRUMENTS.values():
            assert isinstance(inst.asset_class, AssetClass)

    def test_stockholm_instruments_have_no_alpaca_fallback(self):
        # Alpaca cannot see Stockholm listings. Claiming a fallback that does
        # not exist would mask a session outage instead of surfacing it.
        for inst in INSTRUMENTS.values():
            if inst.venue == "STO":
                assert inst.alpaca is None
                assert not inst.has_fallback

    def test_us_equities_have_a_fallback(self):
        for inst in INSTRUMENTS.values():
            if inst.venue == "US":
                assert inst.has_fallback

    def test_certificates_and_warrants_are_sek(self):
        for inst in INSTRUMENTS.values():
            if inst.asset_class in (AssetClass.CERTIFICATE, AssetClass.WARRANT):
                assert inst.currency == "SEK"

    def test_unknown_key_error_is_actionable(self):
        with pytest.raises(KeyError, match="pinned deliberately"):
            by_key("NOPE")

    def test_by_orderbook_roundtrip(self):
        for inst in INSTRUMENTS.values():
            assert by_orderbook(inst.avanza_ob).key == inst.key


class TestGlobalTickerSetsUntouched:
    """P1-4: importing this package must not mutate Tier-1 ticker registries.

    Adding these names to STOCK_SYMBOLS would exhaust the 25/day Alpha Vantage
    quota via alpha_vantage.py:238 and silently stale Tier-1 fundamentals.
    """

    def test_import_does_not_touch_ticker_sets(self):
        from portfolio import tickers

        before = (
            set(tickers.CRYPTO_SYMBOLS),
            set(tickers.METALS_SYMBOLS),
            set(tickers.STOCK_SYMBOLS),
            set(tickers.SYMBOLS),
        )
        import importlib

        import portfolio.swedbank

        importlib.reload(portfolio.swedbank)
        after = (
            set(tickers.CRYPTO_SYMBOLS),
            set(tickers.METALS_SYMBOLS),
            set(tickers.STOCK_SYMBOLS),
            set(tickers.SYMBOLS),
        )
        assert before == after

    def test_swedbank_keys_are_not_in_tier1_symbols(self):
        from portfolio.tickers import SYMBOLS

        overlap = set(INSTRUMENTS) & set(SYMBOLS)
        assert not overlap, f"would inflate Tier-1 loop: {overlap}"


@pytest.mark.integration
class TestPinnedIdsStillResolve:
    """P0-2: a wrong orderbook ID mis-prices a position AND misdirects a real
    manual order. Verify upstream has not moved any of them."""

    def test_each_pinned_id_resolves_to_expected_instrument(self):
        """Compare the NAME upstream reports against the name we pinned.

        The previous version of this test only asserted the quote endpoint
        returned something non-empty — so swapping NVDA's orderbook ID for any
        other valid security would have passed while silently valuing NVDA at
        the wrong price. Checking identity is the entire point.
        """
        import os

        if os.environ.get("PF_LIVE_AVANZA_TESTS") != "1":
            pytest.skip("set PF_LIVE_AVANZA_TESTS=1 to hit the live Avanza session")

        from portfolio.avanza_session import api_get, verify_session

        if not verify_session():
            pytest.skip("no live Avanza session")

        def _norm(x):
            return "".join(ch for ch in (x or "").lower() if ch.isalnum())

        mismatches = []
        for inst in INSTRUMENTS.values():
            detail = None
            for typ in ("stock", "certificate", "warrant", "exchange_traded_fund"):
                try:
                    detail = api_get(f"/_api/market-guide/{typ}/{inst.avanza_ob}")
                except Exception:
                    detail = None
                if detail:
                    break
            if not detail:
                mismatches.append((inst.key, inst.avanza_ob, "no instrument detail"))
                continue
            upstream = detail.get("name") or detail.get("orderbook", {}).get("name")
            a, b = _norm(upstream), _norm(inst.name)
            if not (a and b and (a.startswith(b) or b.startswith(a) or a == b)):
                mismatches.append((inst.key, inst.avanza_ob, f"{upstream!r} != {inst.name!r}"))
        assert not mismatches, f"pinned orderbook ids no longer match: {mismatches}"
