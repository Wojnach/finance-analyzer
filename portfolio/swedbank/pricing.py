"""Pricing for the Swedbank book — Avanza primary, Alpaca fallback, honest staleness.

Design constraints, each from a premortem finding:

* **Sequential only, never concurrent** (P2-6). A full 26-instrument sweep measures
  1.5s sequential, a 2.5% duty cycle at 60s. The real-money metals loop shares this
  Avanza session, so there is no performance justification for concurrency and every
  reason to avoid contending for the Playwright context.
* **Session-error handling.** CAVEAT, corrected 2026-07-31 after review: this module
  calls `avanza_session.api_get`, which internally performs browser teardown/relaunch
  recovery and consecutive-failure escalation. An earlier version of this docstring
  claimed we "never invoke browser recovery" — that was false. What we actually do is
  bound the damage: `SESSION_FAILURE_ABORT` consecutive failures abort the remaining
  sweep instead of hammering all 26 instruments through a dead session, which is what
  would otherwise trigger repeated context relaunches and critical/Telegram escalation
  while the metals loop is managing real orders.
  TODO: MANUAL REVIEW — the real fix is a read-only market-data client that shares no
  browser context with the trading path.
* **Read-only.** `api_get` only — enforced by tests/test_swedbank_no_trading.py.
* **Mark at mid when `last` falls outside bid/ask** (P1-3). Thin instruments carry
  hours-stale prints; one warrant measured +4.04% above mid on 15 units traded.
* **Never present a stale price as live.** Every quote carries source, age and the
  reason it was marked the way it was.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field

from portfolio.swedbank.instruments import INSTRUMENTS

logger = logging.getLogger("portfolio.swedbank.pricing")

STALE_LAST_TOL = 0.005
STALE_QUOTE_S = 1800
SESSION_FAILURE_ABORT = 3
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
    # Reject non-positive and non-finite marks. Only `None` was rejected before,
    # so a suspended instrument quoting last=0 with no bid/ask produced a value of
    # 0 — which the dashboard renders as a clean -100% loss rather than "unpriced".
    if mark is None or not math.isfinite(mark) or mark <= 0:
        raise ValueError(
            f"{inst.key}: no usable price (last={last!r} bid={bid!r} ask={ask!r})"
        )

    age = None
    if updated:
        age = max(0.0, (now_ms - float(updated)) / 1000.0)

    # A quote whose print is hours old, or which the venue itself does not call
    # real-time, is not live data no matter how well-formed it looks.
    quote_degraded = False
    quote_note = "marked at mid: last outside bid/ask" if stale else None
    if raw.get("isRealTime") is False:
        quote_degraded = True
        quote_note = "venue reports quote is not real-time"
    elif age is not None and age > STALE_QUOTE_S:
        quote_degraded = True
        quote_note = f"quote is {age:.0f}s old"

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
        degraded=quote_degraded,
        note=quote_note,
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
    # Distinguish None ("all") from an explicitly empty list ("nothing"). An
    # empty book previously swept all 26 instruments every cycle for no reason.
    keys = list(INSTRUMENTS if keys is None else keys)
    cache = cache or {}
    started = time.time()
    fetch = quote_fn or _avanza_quote_fn()
    sweep_result = PriceSweep(swept_at=started)

    consecutive_failures = 0
    session_dead = False

    for key in keys:
        inst = INSTRUMENTS[key]
        if session_dead:
            # Stop hitting a dead session. Continuing through all 26 instruments
            # is what drives repeated browser relaunches inside api_get and the
            # consecutive-failure escalation, precisely while the metals loop may
            # be managing real orders.
            sweep_result.errors[key] = "skipped: avanza session unavailable"
            _serve_from_cache_or_error(sweep_result, key, cache, now_ms, fallback_fn, inst)
            continue
        try:
            raw = fetch(inst.avanza_ob)
            if not raw:
                raise ValueError("empty quote payload")
            sweep_result.quotes[key] = build_quote(inst, raw, now_ms=now_ms)
            consecutive_failures = 0
            continue
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            consecutive_failures += 1
            if consecutive_failures >= SESSION_FAILURE_ABORT:
                session_dead = True
                logger.warning(
                    "swedbank: %d consecutive avanza failures — treating session as "
                    "down and aborting the remaining sweep (%s)",
                    consecutive_failures,
                    reason,
                )
            else:
                logger.warning(
                    "swedbank: avanza quote failed for %s (%s)", key, reason
                )

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
            # A malformed cache entry must not crash the sweep. This branch only
            # runs when the live sources have ALREADY failed, so an unguarded
            # TypeError here would turn "degraded but still serving last-good
            # prices" into a hard crash at exactly the moment the fallback
            # exists to help — and the bad cache persists on disk, so the loop
            # would crashloop every cycle.
            try:
                q = Quote(
                    **{k: v for k, v in cached.items() if k in Quote.__annotations__}
                )
                q.degraded = True
                q.source = f"{q.source}:cached"
                q.note = "no live source; last-good price"
                if q.as_of_ms:
                    now = now_ms if now_ms is not None else int(time.time() * 1000)
                    q.age_s = max(0.0, (now - q.as_of_ms) / 1000.0)
            except Exception as exc:
                logger.warning("swedbank: unusable cache entry for %s: %s", key, exc)
            else:
                sweep_result.quotes[key] = q
                sweep_result.errors[key] = "no live source; served last-good"
                continue

        sweep_result.errors[key] = "no price available from any source"

    if fx_fn is not None:
        rate, err = resolve_fx(fx_fn)
        if rate is not None:
            sweep_result.fx["USDSEK"] = rate
        if err:
            sweep_result.errors["__fx__"] = err

    sweep_result.duration_s = time.time() - started
    return sweep_result


def _serve_from_cache_or_error(sweep_result, key, cache, now_ms, fallback_fn, inst):
    """Fallback chain used when the live Avanza path is skipped or failed."""
    try:
        fb = (fallback_fn or _alpaca_fallback)(inst)
    except Exception as exc:
        fb = None
        logger.warning("swedbank: alpaca fallback failed for %s: %s", key, exc)
    if fb is not None:
        sweep_result.quotes[key] = fb
        return
    cached = cache.get(key)
    if not cached:
        return
    try:
        q = Quote(**{k: v for k, v in cached.items() if k in Quote.__annotations__})
        q.degraded = True
        q.source = f"{q.source}:cached"
        q.note = "no live source; last-good price"
        if q.as_of_ms:
            now = now_ms if now_ms is not None else int(time.time() * 1000)
            q.age_s = max(0.0, (now - q.as_of_ms) / 1000.0)
    except Exception as exc:
        logger.warning("swedbank: unusable cache entry for %s: %s", key, exc)
        return
    sweep_result.quotes[key] = q


def resolve_fx(fx_fn):
    """Resolve USD/SEK, refusing the silent hard-coded fallback.

    `portfolio.fx_rates.fetch_usd_sek()` does NOT raise when the upstream API
    fails: if its sanity check fails and no cached rate exists it returns
    `FX_RATE_FALLBACK` (10.50). That is ~9.5% above a realistic rate and would
    overstate every USD position — roughly 70% of this book — with no error, no
    exception and a green freshness banner.

    Valuation must never silently use a placeholder. Returns (rate, error): a
    rate of None means USD positions become `unpriced` rather than mis-valued.
    """
    try:
        raw = fx_fn()
    except Exception as exc:
        return None, f"USD/SEK unavailable: {exc}"
    if raw is None:
        return None, "USD/SEK unavailable: source returned None"
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        return None, f"USD/SEK unusable: {raw!r}"
    if not math.isfinite(rate) or rate <= 0:
        return None, f"USD/SEK not a usable rate: {rate!r}"
    try:
        from portfolio.fx_rates import FX_RATE_FALLBACK
    except Exception:
        FX_RATE_FALLBACK = None
    if FX_RATE_FALLBACK is not None and rate == float(FX_RATE_FALLBACK):
        return None, (
            f"USD/SEK returned the hard-coded fallback ({rate}) — the live "
            f"source is down and no cached rate exists. Refusing to value USD "
            f"positions at a placeholder."
        )
    return rate, None


def value_holding(qty, quote, fx, base="SEK"):
    """Convert a position to base currency at the marked price."""
    rate = 1.0 if quote.currency == base else fx
    if rate is None:
        raise ValueError(f"{quote.key}: no FX rate for {quote.currency}->{base}")
    return qty * quote.mark * rate
