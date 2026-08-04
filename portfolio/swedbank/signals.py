"""Signals and trajectories for the Swedbank universe.

Reuses Layer 1's machinery without joining Layer 1:

* `signal_engine.generate_signal(ind, ticker=...)` takes the ticker as a plain
  string key. Every asset-class branch inside it is a set-membership test with a
  safe default, so an unregistered ticker works today and cannot perturb Tier-1.
* Indicators come from `indicators.compute_indicators`, which is pure OHLCV math
  with no registry dependency.
* Trajectories come from `price_targets.compute_targets`, which is genuinely
  instrument-agnostic. The batch wrapper `compute_all_targets` is NOT used: it
  hardcodes `hours_to_us_close()` for every non-24h ticker, which is simply the
  wrong session for a Stockholm listing.

Three things we deliberately do NOT do, each with a specific reason:

1. **Never add these tickers to `tickers.STOCK_SYMBOLS`.** That set is iterated
   by `alpha_vantage.py:238` against a hard 25-requests/day budget and by
   `earnings_calendar.py`; 19 extra names would exhaust the quota and silently
   stale Tier-1's fundamentals. It also drives NYSE-hours GPU gating, which is
   the wrong session for the Stockholm half of this universe.
2. **Never call the sentiment vote path.** `signal_engine.flush_sentiment_state()`
   rewrites `data/sentiment_state.json` as a whole-dict overwrite from a
   per-process in-memory copy. A second process flushing it would clobber
   whatever Layer 1 wrote for its own tickers — last-write-wins data loss.
3. **Never write to `data/signal_log.jsonl`.** `accuracy_stats.signal_accuracy()`
   blends every ticker in that store into one global per-signal accuracy figure
   that Tier-1 falls back on. Our rows would also compete for slots in its
   50k-row tail window, evicting real history. We keep our own log.
"""

from __future__ import annotations

import datetime
import logging

from portfolio.swedbank import ohlcv
from portfolio.swedbank.instruments import INSTRUMENTS, AssetClass

logger = logging.getLogger("portfolio.swedbank.signals")

SIGNAL_LOG = "data/swedbank_signal_log.jsonl"

# Signals that cannot mean anything for this universe. The engine already
# excludes crypto-only and metals-only families for a non-crypto/non-metal
# ticker; these are the ones it would happily run and which would be noise.
_UNINFORMATIVE = {
    # CFTC commercial/speculative positioning has no per-equity series.
    "cot_positioning",
    # NewsAPI headline matching keys off company names — empty for the
    # Stockholm certificates and warrants.
    "news_event",
}

# Leveraged products track an underlying; indicators computed on the
# certificate's own thin, decaying series are misleading. Mirrors the documented
# Tier-3 pattern ("Avanza price + underlying's signals").
UNDERLYING = {
    "XBT-BTC": "BTC-USD",
    "XBT-ETH": "ETH-USD",
    "MINI-TSMC": "TSM",
}


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def applicable_for(inst):
    """Signals worth running for this instrument, from its explicit asset_class.

    Deliberately parameterised on the instrument rather than reading the global
    ticker sets — see the module docstring, point 1.
    """
    from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES

    crypto_only = {"futures_flow", "funding", "crypto_macro", "onchain"}
    metals_only = {"metals_cross_asset"}
    non_stock = {"orderbook_flow"}

    out = []
    for sig in SIGNAL_NAMES:
        if sig in DISABLED_SIGNALS or sig in _UNINFORMATIVE:
            continue
        if sig in crypto_only or sig in metals_only:
            continue
        # An equity IS a stock, so non-stock-only signals are excluded. Relying
        # on global-set membership here is exactly the bug that lets
        # orderbook_flow leak onto equities.
        if sig in non_stock and inst.asset_class == AssetClass.EQUITY:
            continue
        out.append(sig)
    return out


def evaluate(
    inst, horizon="1d", chart_fn=None, config=None, alpaca_fn=None, ticker_fn=None
):
    """Signals + trajectory for one instrument.

    Returns a dict, or one carrying `error` — never a neutral/HOLD verdict
    fabricated from missing data. A monitoring page that shows HOLD when it
    actually has no data is worse than one that shows nothing.
    """
    from portfolio.indicators import compute_indicators

    key = inst.key
    signal_ticker = UNDERLYING.get(key, key)
    base = {
        "key": key,
        "name": inst.name,
        "asset_class": inst.asset_class.value,
        "horizon": horizon,
        "signal_ticker": signal_ticker,
        "computed_at": _now_iso(),
    }
    if signal_ticker != key:
        base["note"] = (
            f"signals computed on underlying {signal_ticker}; leverage/decay make "
            f"the certificate's own series unreliable for indicators"
        )

    # An UNDERLYING that is not itself a pinned instrument (BTC-USD, ETH-USD)
    # has no orderbook ID, so it cannot use the Avanza chart path. Previously
    # `INSTRUMENTS.get(signal_ticker, inst)` silently fell back to the
    # CERTIFICATE — computing indicators on exactly the thin, decaying SEK series
    # the underlying mapping exists to avoid, while the payload still claimed the
    # underlying was used. Route those through price_source instead, and never
    # self-reference.
    # A watchlist instrument is not pinned in INSTRUMENTS but carries its own
    # orderbook ID, so it can use the Avanza chart path directly. Only when the
    # signal ticker is a DIFFERENT unpinned symbol (an underlying like BTC-USD)
    # must it route through price_source instead.
    src_inst = INSTRUMENTS.get(signal_ticker)
    if src_inst is None and signal_ticker == key:
        src_inst = inst
    try:
        if src_inst is not None:
            df, source = ohlcv.fetch(
                src_inst, horizon=horizon, chart_fn=chart_fn, alpaca_fn=alpaca_fn
            )
        elif signal_ticker != key:
            df, source = (ticker_fn or ohlcv.fetch_by_ticker)(
                signal_ticker, horizon=horizon
            )
        else:
            return {**base, "error": f"{key}: no pinned instrument and no source"}
    except Exception as exc:
        return {**base, "error": f"no OHLCV: {exc}"}
    base["ohlcv_source"] = source
    base["bars"] = int(len(df))

    ind = compute_indicators(df, horizon=None)
    if not ind:
        return {**base, "error": f"insufficient history ({len(df)} bars)"}

    try:
        from portfolio.signal_engine import generate_signal

        # generate_signal returns a 3-TUPLE (action, confidence, extra), not a
        # dict — see the canonical caller at portfolio/main.py:512.
        action, confidence, extra = generate_signal(
            ind, ticker=signal_ticker, config=config, df=df
        )
    except Exception as exc:
        logger.warning("swedbank signals: generate_signal failed for %s: %s", key, exc)
        return {**base, "error": f"signal engine failed: {exc}"}

    extra = extra or {}
    action = action or "HOLD"
    confidence = float(confidence or 0.0)
    # Votes are only exposed under the private `_raw_votes` key (pre-gate,
    # shadow votes merged in) — there is no public "votes"/"signals" entry, and
    # reading one that does not exist renders an empty vote panel forever.
    votes = extra.get("_raw_votes") or {}

    out = {
        **base,
        "action": action,
        "confidence": confidence,
        "regime": extra.get("_regime") or "",
        "votes": {k: v for k, v in votes.items() if v and v != "HOLD"},
        "vote_counts": {
            "buy": sum(1 for v in votes.values() if v == "BUY"),
            "sell": sum(1 for v in votes.values() if v == "SELL"),
            "hold": sum(1 for v in votes.values() if v == "HOLD"),
        },
        "applicable": len(applicable_for(inst)),
        "price": float(df["close"].iloc[-1]),
        "atr_pct": extra.get("atr_pct") or ind.get("atr_pct"),
    }
    out["trajectory"] = _trajectory(inst, out, ind, extra)
    out["news"] = _news_block(key, signal_ticker)
    return out


def _news_block(key, signal_ticker=None):
    """Display-only headline scores. Imported here, not at module scope, because
    news.py reads UNDERLYING from this module — a top-level import would cycle.

    News never votes: `out["action"]` above is already final. Attaching an
    uncalibrated keyword score to consensus would move decisions on nothing but
    word counts, and calibration needs realized outcomes we do not have yet.
    """
    try:
        from portfolio.swedbank import news as newsmod

        return newsmod.fetch_for(key, ticker=signal_ticker)
    except Exception as exc:
        logger.warning("swedbank news block failed for %s: %s", key, exc)
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _hours_remaining(inst):
    """Hours until this instrument's session closes, DST-correct.

    Two bugs this replaces, both silent:

    * Hardcoded UTC constants (15.5 STO / 20.0 US) are only right during summer
      time. Stockholm closes 17:30 local = 15:30 UTC in CEST but 16:30 UTC in
      CET; New York 16:00 local = 20:00 UTC in EDT but 21:00 in EST. So every
      projection was mis-scaled by an hour for roughly five months a year.
    * Wrapping past midnight returned up to 24h, so with the market shut the
      trajectory projected a full day of volatility as though it were an
      intraday move.

    Returns (hours, market_open). When the market is closed we report the hours
    to the NEXT close but flag it, so a caller can say so rather than presenting
    an overnight projection as an intraday one.
    """
    from zoneinfo import ZoneInfo

    tz, close_h, close_m, open_h, open_m = (
        (ZoneInfo("Europe/Stockholm"), 17, 30, 9, 0)
        if inst.venue == "STO"
        else (ZoneInfo("America/New_York"), 16, 0, 9, 30)
    )
    now = datetime.datetime.now(tz)
    close = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    open_t = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)

    market_open = open_t <= now < close and now.weekday() < 5
    if now >= close or now.weekday() >= 5:
        # Roll to the next weekday's close.
        close += datetime.timedelta(days=1)
        while close.weekday() >= 5:
            close += datetime.timedelta(days=1)

    hours = (close - now).total_seconds() / 3600.0
    if not market_open:
        # With the market shut, wall-clock to the next close can be 60h+ over a
        # weekend. Feeding that to the projection scales it by sqrt(60) and
        # yields a ladder so wide it means nothing. Cap at one session so the
        # numbers describe the NEXT session, which is what a closed-market
        # projection can honestly claim.
        session_len = 8.5 if inst.venue == "STO" else 6.5
        hours = min(hours, session_len)
    return max(0.25, hours), market_open


def _trajectory(inst, sig, ind, extra):
    """Forward price targets. Instrument-agnostic core, per-venue session."""
    try:
        from portfolio.price_targets import compute_targets
    except Exception as exc:
        return {"error": f"price_targets unavailable: {exc}"}

    action = sig.get("action") or "HOLD"
    # price_targets branches on lowercase "buy"/"sell" (price_targets.py:93,137,
    # 281-287) — "LONG"/"SHORT" matched NEITHER, so every comparison fell to the
    # else branch and the whole projection ran with inverted directional logic
    # while still returning plausible targets. Verified against the callee.
    side = "sell" if action == "SELL" else "buy"
    atr_pct = sig.get("atr_pct")
    if not atr_pct or atr_pct <= 0:
        return {"error": "no ATR — cannot project a range"}

    hours_left, market_open = _hours_remaining(inst)
    is_24h = sig["signal_ticker"].endswith("-USD")
    if not is_24h and inst.venue == "STO":
        # price_targets._year_fraction hardcodes 6.5h as one trading day, but a
        # Stockholm session is 8.5h and one daily Avanza bar spans that whole
        # session — so one full session must equal exactly ONE trading day of
        # variance, not 8.5/6.5 = 1.308. Left uncorrected this inflated every
        # STO band by sqrt(1.308) = 1.143, permanently. Convert to
        # US-equivalent trading hours.
        hours_left = hours_left * (6.5 / 8.5)
    conf = max(0.0, min(1.0, float(sig.get("confidence") or 0.0)))
    # Map consensus to a directional probability, centred on 0.5 for HOLD so a
    # no-opinion verdict produces a symmetric range rather than a fake edge.
    p_up = 0.5 + (
        conf / 2.0 if action == "BUY" else -conf / 2.0 if action == "SELL" else 0.0
    )
    p_up = max(0.05, min(0.95, p_up))

    try:
        t = compute_targets(
            ticker=sig["signal_ticker"],
            side=side,
            price_usd=float(sig["price"]),
            atr_pct=float(atr_pct),
            p_up=p_up,
            hours_remaining=hours_left,
            indicators=ind,
            extra=extra,
            is_24h=is_24h,
            regime=sig.get("regime") or "",
            n_paths=4000,
        )
    except Exception as exc:
        logger.warning("swedbank trajectory failed for %s: %s", inst.key, exc)
        return {"error": f"trajectory failed: {exc}"}

    # compute_targets returns ticker/side/price_usd/hours_remaining/extremes/
    # targets/recommended — there is no expected_move_pct key. Derive the
    # projected move from `extremes` rather than inventing a field.
    t = t or {}
    extremes = t.get("extremes") or {}
    price = float(sig["price"]) or 0.0
    # `extremes` is a PERCENTILE dict (p10/p25/p50/p75/p90), not high/low — so
    # the earlier high/low lookup always yielded None. p10..p90 is the honest
    # 80% projection band.
    move_pct = None
    lo, hi = extremes.get("p10"), extremes.get("p90")
    if price and hi is not None and lo is not None:
        move_pct = round((float(hi) - float(lo)) / price * 100.0, 2)
    return {
        "side": side,
        "p_up": round(p_up, 4),
        "hours_remaining": round(hours_left, 2),
        "market_open": market_open,
        "targets": t.get("targets") or [],
        "recommended": t.get("recommended"),
        "extremes": extremes,
        "projected_range_pct": move_pct,  # p10..p90 band, % of spot
        "atr_pct": atr_pct,
    }


def evaluate_universe(
    keys=None, horizon="1d", chart_fn=None, config=None, alpaca_fn=None, ticker_fn=None
):
    """Sequentially evaluate the universe. Sequential for the same reason
    pricing.sweep is: the real-money metals loop shares this Avanza session."""
    keys = list(INSTRUMENTS if keys is None else keys)
    # No result cache. A previous version memoised whole result rows per
    # underlying, which could never hit in production (underlyings cannot be
    # book keys — book.py validates against INSTRUMENTS — and no two keys share
    # one) while silently leaking asset_class, the underlying note and the other
    # instrument's venue-specific hours between rows the moment a caller passed
    # unsorted keys. Dead code whose only reachable behaviour is confident-wrong
    # output is exactly this subsystem's enemy. If dedup is ever needed, memoise
    # (df, source) per signal_ticker — cache the DATA, never the row.
    out = {}
    for key in keys:
        out[key] = evaluate(
            INSTRUMENTS[key],
            horizon=horizon,
            chart_fn=chart_fn,
            config=config,
            alpaca_fn=alpaca_fn,
            ticker_fn=ticker_fn,
        )
    return out


def log_snapshot(results, path=SIGNAL_LOG):
    """Append to our OWN log, never data/signal_log.jsonl (docstring point 3)."""
    from portfolio.file_utils import atomic_append_jsonl

    entry = {
        "ts": _now_iso(),
        "tickers": {
            k: {
                "action": v.get("action"),
                "confidence": v.get("confidence"),
                "regime": v.get("regime"),
                "error": v.get("error"),
            }
            for k, v in results.items()
        },
    }
    try:
        atomic_append_jsonl(path, entry)
    except Exception as exc:
        logger.warning("swedbank signal log append failed: %s", exc)
    return entry
