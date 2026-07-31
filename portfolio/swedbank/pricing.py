"""Pricing for the Swedbank book — Avanza primary, Alpaca fallback, honest staleness.

Design constraints, each from a premortem finding:

* **Sequential only, never concurrent** (P2-6). A full 26-instrument sweep measures
  1.5s sequential, a 2.5% duty cycle at 60s. The real-money metals loop shares this
  Avanza session, so there is no performance justification for concurrency and every
  reason to avoid contending for the Playwright context.
* **Never invoke browser recovery.** On session error this module degrades and backs
  off, leaving recovery to the loops that actually trade.
* **Read-only.** `api_get` only — enforced by tests/test_swedbank_no_trading.py.
* **Mark at mid when `last` falls outside bid/ask** (P1-3). Thin instruments carry
  hours-stale prints; one warrant measured +4.04% above mid on 15 units traded.
* **Never present a stale price as live.** Every quote carries source, age and the
  reason it was marked the way it was.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from portfolio.swedbank.instruments import INSTRUMENTS

logger = logging.getLogger("portfolio.swedbank.pricing")

STALE_LAST_TOL = 0.005
CACHE_PATH = "data/swedbank_prices.json"


@dataclass
class Quote:
    key: str
    mark: float
    currency: str
    source: str
    mark_basis: str
    as_of_ms: int | None = None
    age_s: float | None = None
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    mid: float | None = None
    spread_pct: float | None = None
    volume: float | None = None
    is_real_time: bool | None = None
    stale_last: bool = False
    degraded: bool = False
    note: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class PriceSweep:
    quotes: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    fx: dict = field(default_factory=dict)
    swept_at: float = 0.0
    duration_s: float = 0.0

    @property
    def ok(self):
        return not self.errors

    def to_dict(self):
        return {
            "quotes": {k: q.to_dict() for k, q in self.quotes.items()},
            "errors": dict(self.errors),
            "fx": dict(self.fx),
            "swept_at": self.swept_at,
            "duration_s": self.duration_s,
        }


def _avanza_quote_fn():
    from portfolio.avanza_session import api_get

    def _fn(ob):
        return api_get(f"/_api/market-guide/stock/{ob}/quote")

    return _fn


def build_quote(inst, raw, now_ms=None):
    """Turn a raw Avanza quote payload into a marked Quote.

    Marking rule: use `last` when it sits inside the bid/ask, otherwise use mid
    and flag it. A `last` outside the spread means the print is old — the book
    would otherwise be valued at a price nobody is currently offering.
    """
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    bid = raw.get("buy")
    ask = raw.get("sell")
    last = raw.get("last")
    updated = raw.get("updated") or raw.get("timeOfLast")

    mid = None
    spread_pct = None
    if bid and ask and bid > 0 and ask > 0:
        mid = (bid + ask) / 2.0
        if mid > 0:
            spread_pct = (ask - bid) / mid * 100.0

    mark, basis, stale = last, "last", False
    if mid is not None:
        if last is None or last <= 0:
            mark, basis = mid, "mid"
        elif not (min(bid, ask) <= last <= max(bid, ask)):
            if abs(last - mid) / mid > STALE_LAST_TOL:
                mark, basis, stale = mid, "mid", True
    if mark is None:
        raise ValueError(f"{inst.key}: quote carries neither usable last nor bid/ask")

    age = None
    if updated:
        age = max(0.0, (now_ms - float(updated)) / 1000.0)

    return Quote(
        key=inst.key,
        mark=float(mark),
        currency=inst.currency,
        source="avanza",
        mark_basis=basis,
        as_of_ms=int(updated) if updated else None,
        age_s=age,
        bid=bid,
        ask=ask,
        last=last,
        mid=mid,
        spread_pct=spread_pct,
        volume=raw.get("totalVolumeTraded"),
        is_real_time=raw.get("isRealTime"),
        stale_last=stale,
        note="marked at mid: last outside bid/ask" if stale else None,
    )


def _alpaca_fallback(inst):
    """Last-resort quote for US names when the Avanza session is unavailable.

    Alpaca is IEX-only and carries no bid/ask, so the resulting Quote is flagged
    degraded and leaves the spread fields empty rather than implying they are
    zero or unknown-but-fine.
    """
    if not inst.has_fallback:
        return None
    from portfolio.price_source import fetch_klines

    df = fetch_klines(inst.alpaca, interval="1d", limit=2)
    if df is None or df.empty:
        return None
    return Quote(
        key=inst.key,
        mark=float(df["close"].iloc[-1]),
        currency=inst.currency,
        source="alpaca:fallback",
        mark_basis="close",
        degraded=True,
        note="Avanza unavailable; Alpaca IEX close, no bid/ask available",
    )


def sweep(
    keys=None, quote_fn=None, now_ms=None, cache=None, fx_fn=None, fallback_fn=None
):
    """Fetch quotes for the given instruments, sequentially.

    Returns a PriceSweep. Individual failures are collected rather than raised —
    a partial book that is honest about which rows are stale beats no book.
    Degradation order per instrument: Avanza -> Alpaca (US only) -> cached
    last-good -> recorded error.
    """
    keys = list(keys or INSTRUMENTS)
    cache = cache or {}
    started = time.time()
    fetch = quote_fn or _avanza_quote_fn()
    sweep_result = PriceSweep(swept_at=started)

    for key in keys:
        inst = INSTRUMENTS[key]
        try:
            raw = fetch(inst.avanza_ob)
            if not raw:
                raise ValueError("empty quote payload")
            sweep_result.quotes[key] = build_quote(inst, raw, now_ms=now_ms)
            continue
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("swedbank: avanza quote failed for %s (%s)", key, reason)

        try:
            fb = (fallback_fn or _alpaca_fallback)(inst)
        except Exception as exc:
            fb = None
            logger.warning("swedbank: alpaca fallback failed for %s: %s", key, exc)
        if fb is not None:
            sweep_result.quotes[key] = fb
            sweep_result.errors[key] = "avanza unavailable, served from alpaca"
            continue

        cached = cache.get(key)
        if cached:
            q = Quote(**{k: v for k, v in cached.items() if k in Quote.__annotations__})
            q.degraded = True
            q.source = f"{q.source}:cached"
            q.note = "no live source; last-good price"
            if q.as_of_ms:
                now = now_ms if now_ms is not None else int(time.time() * 1000)
                q.age_s = max(0.0, (now - q.as_of_ms) / 1000.0)
            sweep_result.quotes[key] = q
            sweep_result.errors[key] = "no live source; served last-good"
            continue

        sweep_result.errors[key] = "no price available from any source"

    if fx_fn is not None:
        try:
            sweep_result.fx["USDSEK"] = float(fx_fn())
        except Exception as exc:
            # A silently-stale FX rate mis-values ~70% of the book by whatever
            # the drift is, with no other symptom. Record it as an error.
            sweep_result.errors["__fx__"] = f"USD/SEK unavailable: {exc}"

    sweep_result.duration_s = time.time() - started
    return sweep_result


def value_holding(qty, quote, fx, base="SEK"):
    """Convert a position to base currency at the marked price."""
    rate = 1.0 if quote.currency == base else fx
    if rate is None:
        raise ValueError(f"{quote.key}: no FX rate for {quote.currency}->{base}")
    return qty * quote.mark * rate
