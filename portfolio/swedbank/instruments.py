"""Pinned instrument table — the single source of truth for this subsystem.

Orderbook IDs are PINNED, never resolved by runtime search. During exploration
(2026-07-31) naive "first STOCK search hit" resolution returned the wrong
instrument for 2 of 19 US names: one query missed entirely because it used the
legal name rather than the ticker, and one matched the wrong SHARE CLASS
(Class C returned for a Class A holding — near-identical name, near-identical
price, invisible on inspection). A third name has a decoy second listing at a
different orderbook ID.

The orderbook ID is both the pricing key and the deep-link the operator clicks
to place a real order by hand, so a wrong ID mis-values a position AND routes a
real order to the wrong security. `tests/test_swedbank_instruments.py` asserts
every pinned ID still resolves to its expected name, so an upstream change fails
loudly instead of silently repricing.

`asset_class` is carried explicitly rather than inferred from
`tickers.STOCK_SYMBOLS`/`CRYPTO_SYMBOLS`/`METALS_SYMBOLS`. Adding these names to
those global sets would exhaust the 25/day Alpha Vantage quota via
`alpha_vantage.py:238` and starve Tier-1; leaving them out makes
`signal_engine._compute_applicable_count` read them as non-stocks and apply
`orderbook_flow` to equities. Carrying the field sidesteps both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetClass(str, Enum):
    EQUITY = "equity"
    CERTIFICATE = "certificate"
    WARRANT = "warrant"


@dataclass(frozen=True)
class Instrument:
    key: str
    name: str
    asset_class: AssetClass
    currency: str
    avanza_ob: str
    alpaca: str | None = None
    venue: str = "US"

    @property
    def has_fallback(self) -> bool:
        return self.alpaca is not None


_DERIVE = object()  # sentinel: distinguishes "not supplied" from an explicit None


def _eq(key, name, ob, currency="USD", alpaca=_DERIVE, venue="US"):
    # alpaca=None must mean "no fallback exists", not "derive it from the key".
    # Conflating the two made the Stockholm equities advertise an Alpaca
    # fallback that cannot work, which would mask an Avanza outage rather than
    # degrading honestly to last-good-price.
    return Instrument(
        key=key,
        name=name,
        asset_class=AssetClass.EQUITY,
        currency=currency,
        avanza_ob=ob,
        alpaca=key if alpaca is _DERIVE else alpaca,
        venue=venue,
    )


_US = [
    _eq("NVDA", "NVIDIA", "4478"),
    _eq("MU", "Micron Technology", "214533"),
    _eq("AMD", "Advanced Micro Devices", "529720"),
    _eq("TSM", "Taiwan Semicond Mfg Co", "34911"),
    _eq("WDC", "Western Digital", "353285"),
    _eq("STX", "Seagate Technology", "150628"),
    _eq("SNDK", "Sandisk", "1968764"),
    # 3844 is the US KLAC listing. A second "KLA" exists at ob 1706880 — a
    # different listing, do not use it. Searching "KLA Corporation" returns
    # zero hits; only the ticker resolves.
    _eq("KLAC", "KLA", "3844"),
    _eq("LRCX", "Lam Research", "3914"),
    _eq("INTC", "Intel", "3658"),
    _eq("AVGO", "Broadcom", "369636"),
    _eq("DELL", "Dell Technologies C", "918953"),
    _eq("NBIS", "Nebius Group", "301310"),
    _eq("CRDO", "Credo Technology", "1337422"),
    _eq("ALAB", "Astera Labs", "1738607"),
    _eq("APLD", "Applied Digital", "1392605"),
    _eq("CRWV", "CoreWeave", "2009607"),
    _eq("AAPL", "Apple", "3323"),
    # Class A (GOOGL). Class C (GOOG) is ob 4457 and is NOT this holding —
    # the two differ by <1% in price and are trivially confused.
    _eq("GOOGL", "Alphabet Inc Class A", "472095"),
]

# Stockholm — no Alpaca fallback exists for any of these. On Avanza session
# loss they degrade to last-good-price with an explicit age stamp.
_STO = [
    _eq("INVE-B", "Investor B", "5247", currency="SEK", alpaca=None, venue="STO"),
    _eq("SAAB-B", "SAAB B", "5401", currency="SEK", alpaca=None, venue="STO"),
    _eq("SEB-C", "SEB C", "5256", currency="SEK", alpaca=None, venue="STO"),
    _eq("BEAMMW-B", "Beammwave B", "1361888", currency="SEK", alpaca=None, venue="STO"),
    Instrument(
        key="XBT-BTC",
        name="CoinShares XBT Provider Bitcoin Tracker One",
        asset_class=AssetClass.CERTIFICATE,
        currency="SEK",
        avanza_ob="563966",
        venue="STO",
    ),
    Instrument(
        key="XBT-ETH",
        name="CoinShares XBT Provider Ether Tracker One",
        asset_class=AssetClass.CERTIFICATE,
        currency="SEK",
        avanza_ob="791709",
        venue="STO",
    ),
    Instrument(
        key="MINI-TSMC",
        name="MINI L TSMC AVA 19",
        asset_class=AssetClass.WARRANT,
        currency="SEK",
        avanza_ob="1586027",
        venue="STO",
    ),
]

INSTRUMENTS: dict[str, Instrument] = {i.key: i for i in (*_US, *_STO)}


def by_key(key: str) -> Instrument:
    try:
        return INSTRUMENTS[key]
    except KeyError:
        raise KeyError(
            f"unknown swedbank instrument {key!r}; "
            f"add it to portfolio/swedbank/instruments.py "
            f"(orderbook IDs are pinned deliberately, never resolved at runtime)"
        ) from None


def by_orderbook(ob: str) -> Instrument:
    for inst in INSTRUMENTS.values():
        if inst.avanza_ob == ob:
            return inst
    raise KeyError(f"no pinned instrument with orderbook id {ob!r}")
