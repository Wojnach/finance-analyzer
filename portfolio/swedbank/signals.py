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


def evaluate(inst, horizon="1d", chart_fn=None, config=None):
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

    src_inst = INSTRUMENTS.get(signal_ticker, inst)
    try:
        df, source = ohlcv.fetch(src_inst, horizon=horizon, chart_fn=chart_fn)
    except Exception as exc:
        return {**base, "error": f"no OHLCV: {exc}"}
    base["ohlcv_source"] = source
    base["bars"] = int(len(df))

    ind = compute_indicators(df, horizon=None)
    if not ind:
        return {**base, "error": f"insufficient history ({len(df)} bars)"}

    try:
        from portfolio.signal_engine import generate_signal

        verdict = generate_signal(ind, ticker=signal_ticker, config=config, df=df)
    except Exception as exc:
        logger.warning("swedbank signals: generate_signal failed for %s: %s", key, exc)
        return {**base, "error": f"signal engine failed: {exc}"}

    action = (verdict or {}).get("action") or "HOLD"
    confidence = float((verdict or {}).get("confidence") or 0.0)
    votes = (verdict or {}).get("votes") or {}
    extra = (verdict or {}).get("extra_info") or {}

    out = {
        **base,
        "action": action,
        "confidence": confidence,
        "regime": extra.get("regime") or (verdict or {}).get("regime") or "",
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
    return out


def _hours_remaining(inst):
    """Hours until this instrument's session closes.

    Stockholm closes 17:30 CET, US 16:00 ET. `price_targets.compute_all_targets`
    hardcodes the US close for every non-24h ticker, which is why we compute this
    ourselves rather than using that wrapper.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    close_utc_hour = 15.5 if inst.venue == "STO" else 20.0
    now_h = now.hour + now.minute / 60.0
    remaining = close_utc_hour - now_h
    if remaining <= 0:
        remaining += 24.0
    return max(0.5, remaining)


def _trajectory(inst, sig, ind, extra):
    """Forward price targets. Instrument-agnostic core, per-venue session."""
    try:
        from portfolio.price_targets import compute_targets
    except Exception as exc:
        return {"error": f"price_targets unavailable: {exc}"}

    action = sig.get("action") or "HOLD"
    side = "LONG" if action == "BUY" else "SHORT" if action == "SELL" else "LONG"
    atr_pct = sig.get("atr_pct")
    if not atr_pct or atr_pct <= 0:
        return {"error": "no ATR — cannot project a range"}

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
            hours_remaining=_hours_remaining(inst),
            indicators=ind,
            extra=extra,
            is_24h=False,
            regime=sig.get("regime") or "",
            n_paths=4000,
        )
    except Exception as exc:
        logger.warning("swedbank trajectory failed for %s: %s", inst.key, exc)
        return {"error": f"trajectory failed: {exc}"}

    return {
        "side": side,
        "p_up": round(p_up, 4),
        "hours_remaining": round(_hours_remaining(inst), 2),
        "targets": (t or {}).get("targets") or [],
        "expected_move_pct": (t or {}).get("expected_move_pct"),
        "atr_pct": atr_pct,
    }


def evaluate_universe(keys=None, horizon="1d", chart_fn=None, config=None):
    """Sequentially evaluate the universe. Sequential for the same reason
    pricing.sweep is: the real-money metals loop shares this Avanza session."""
    keys = list(INSTRUMENTS if keys is None else keys)
    # Certificates share an underlying with each other; evaluating the same
    # underlying repeatedly would triple the Avanza chart calls for no gain.
    cache, out = {}, {}
    for key in keys:
        inst = INSTRUMENTS[key]
        under = UNDERLYING.get(key, key)
        if under in cache and under != key:
            out[key] = {**cache[under], "key": key, "name": inst.name}
            continue
        res = evaluate(inst, horizon=horizon, chart_fn=chart_fn, config=config)
        out[key] = res
        cache[under] = res
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
