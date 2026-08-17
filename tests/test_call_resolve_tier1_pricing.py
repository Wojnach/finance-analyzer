"""call_journal_resolve must be able to price Tier-1 instruments.

CALLS-VERIFY-1D ran on 2026-08-17 and reported:

    Resolved 0/2 due calls. Unpriced, left open: ['XAG-USD', 'XAU-USD']

_price_map only knew two sources: Swedbank pinned INSTRUMENTS (26 equities and
certificates) and watchlist orderbook IDs. Neither contains XAU-USD/XAG-USD —
those are Binance FAPI synthetics — so metals calls could never be scored and
the pickup would have re-deferred forever, silently never calibrating the
judgment layer it exists to calibrate.

portfolio.price_source already prices the whole Tier-1 universe (metals via
binance_fapi, crypto via binance_spot, MSTR via alpaca). It becomes the third
and last tier, after the two Avanza-backed sources so the real-money session
is still preferred where it has a quote.
"""

import sys
import types

import pytest

sys.path.insert(0, "scripts")

from pickups import call_journal_resolve as mod  # noqa: E402


@pytest.fixture
def no_avanza(monkeypatch):
    """Neutralise both Avanza-backed tiers so only the fallback can answer."""
    monkeypatch.setattr(mod, "_swedbank_prices", lambda keys: {}, raising=False)
    monkeypatch.setattr(mod, "_watchlist_prices", lambda root, keys: {}, raising=False)
    return monkeypatch


def _fake_price_source(prices):
    m = types.ModuleType("portfolio.price_source")

    class _DF:
        def __init__(self, v):
            self._v = v

        def __getitem__(self, k):
            return types.SimpleNamespace(iloc=[self._v])

    def fetch_klines(ticker, interval="1h", limit=2, **kw):
        if ticker not in prices:
            raise RuntimeError(f"no source for {ticker}")
        return {"close": [prices[ticker]]}

    m.fetch_klines = fetch_klines
    return m


def test_metals_are_priced_via_price_source(monkeypatch, tmp_path, no_avanza):
    monkeypatch.setitem(
        sys.modules,
        "portfolio.price_source",
        _fake_price_source({"XAU-USD": 4424.30, "XAG-USD": 66.17}),
    )
    out = mod._price_map(tmp_path, ["XAU-USD", "XAG-USD"])
    assert out == {"XAU-USD": pytest.approx(4424.30), "XAG-USD": pytest.approx(66.17)}


def test_unknown_ticker_is_simply_absent_not_an_exception(
    monkeypatch, tmp_path, no_avanza
):
    """A pricing outage must leave a call open, never crash the pickup."""
    monkeypatch.setitem(sys.modules, "portfolio.price_source", _fake_price_source({}))
    assert mod._price_map(tmp_path, ["XAU-USD", "NOPE"]) == {}


def test_avanza_sources_win_over_the_fallback(monkeypatch, tmp_path):
    """The real-money session is preferred wherever it actually has a quote."""
    monkeypatch.setattr(mod, "_swedbank_prices", lambda keys: {"NVDA": 111.0})
    monkeypatch.setattr(mod, "_watchlist_prices", lambda root, keys: {})
    monkeypatch.setitem(
        sys.modules,
        "portfolio.price_source",
        _fake_price_source({"NVDA": 999.0, "XAU-USD": 4424.30}),
    )
    out = mod._price_map(tmp_path, ["NVDA", "XAU-USD"])
    assert out["NVDA"] == 111.0, "fallback overrode a live Avanza mark"
    assert out["XAU-USD"] == pytest.approx(4424.30)


def test_price_source_failure_does_not_break_the_other_tickers(
    monkeypatch, tmp_path, no_avanza
):
    monkeypatch.setitem(
        sys.modules, "portfolio.price_source", _fake_price_source({"XAG-USD": 66.17})
    )
    out = mod._price_map(tmp_path, ["XAU-USD", "XAG-USD"])
    assert out == {"XAG-USD": pytest.approx(66.17)}


def test_price_map_never_raises_when_every_source_is_down(monkeypatch, tmp_path):
    """Docstring contract: 'Never raises: a pricing outage must leave the calls open'."""

    def boom(*a, **kw):
        raise RuntimeError("everything is on fire")

    monkeypatch.setattr(mod, "_swedbank_prices", boom)
    monkeypatch.setattr(mod, "_watchlist_prices", boom)
    broken = types.ModuleType("portfolio.price_source")
    broken.fetch_klines = boom
    monkeypatch.setitem(sys.modules, "portfolio.price_source", broken)

    assert mod._price_map(tmp_path, ["XAU-USD"]) == {}
