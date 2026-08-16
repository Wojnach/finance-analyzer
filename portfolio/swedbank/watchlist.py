"""Avanza watchlist ("Mina bevakningar") tracking — signals for names we don't own.

Endpoint `/_api/watchlist/watchlist` (documented in the unofficial clients,
verified live 2026-08-04: one list, 45 orderbook IDs, some 404-dead). The list
is user-curated in the Avanza app and changes without warning, so it is
re-fetched once a day inside the loop rather than pinned in code — the opposite
of instruments.py, whose IDs are pinned because they price real positions.

Instruments already in the Swedbank book are NOT re-evaluated here; their row
carries a pointer instead. Signals on the same underlying would burn a second
OHLCV fetch per cycle for an identical answer.

Same session discipline as the rest of the package: sequential, read-only,
never trades.
"""

from __future__ import annotations

import datetime
import logging
from functools import partial
import re

from portfolio.swedbank.instruments import INSTRUMENTS, AssetClass, Instrument

logger = logging.getLogger("portfolio.swedbank.watchlist")

WATCHLIST_PATH = "/_api/watchlist/watchlist"
CACHE_FILE = "data/watchlist_instruments.json"
SNAPSHOT_FILE = "data/watchlist_snapshot.json"
SIGNAL_LOG = "data/watchlist_signal_log.jsonl"
REFRESH_SECONDS = 24 * 3600

# Leveraged/tracker products mapped to a researchable underlying, mirroring
# signals.UNDERLYING. Keyed by orderbook ID because these products' names are
# free text. BZ=F routes through price_source's yfinance path (same as oil_loop).
UNDERLYING_BY_OB = {
    "2224675": "MSTR",  # BEAR MSTR X5 SG5
    "2060768": "BZ=F",  # MINI L OLJA AVA 601
    "2510004": "SPCX",  # BULL SPACEX X6 VT4
}

# No signal source exists for these: funds without OHLCV-style series, or
# products whose underlying has no reachable feed. Quotes only.
QUOTE_ONLY_OBS = {
    "86287",  # Pictet-Clean Energy fund
    "2438978",  # Spiltan Europafond fund
    "947022",  # AVA SAMSUNG TRACKER — Samsung has no feed in our stack
}


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _slug(name, ob):
    s = re.sub(r"[^A-Za-z0-9]+", "-", (name or "").strip()).strip("-").upper()
    return s[:24] or f"OB-{ob}"


def fetch_watchlists(api_get_fn=None):
    if api_get_fn is None:
        from portfolio.avanza_session import api_get as api_get_fn
    lists = api_get_fn(WATCHLIST_PATH)
    if not isinstance(lists, list):
        raise RuntimeError(f"watchlist endpoint returned {type(lists).__name__}")
    return lists


def _resolve(ob, api_get_fn):
    g = api_get_fn(f"/_api/market-guide/stock/{ob}")
    listing = g.get("listing") or {}
    return {
        "ob": str(ob),
        "name": g.get("name") or listing.get("shortName") or f"OB-{ob}",
        "ticker": (listing.get("tickerSymbol") or "").strip(),
        "currency": listing.get("currency") or "SEK",
    }


def refresh_cache(api_get_fn=None, resolve_fn=None):
    """Fetch the live watchlist and resolve every orderbook. Returns the cache
    dict that was written. Dead orderbooks (delisted/expired) are recorded, not
    dropped — silently shrinking the list would hide that an instrument the
    user watches stopped existing."""
    from portfolio.file_utils import atomic_write_json

    if api_get_fn is None:
        from portfolio.avanza_session import api_get as api_get_fn
    resolve = resolve_fn or (lambda ob: _resolve(ob, api_get_fn))

    lists = fetch_watchlists(api_get_fn)
    entries, dead = [], []
    seen = set()
    for wl in lists:
        for ob in wl.get("orderbookIds") or []:
            ob = str(ob)
            if ob in seen:
                continue
            seen.add(ob)
            try:
                entries.append({**resolve(ob), "list": wl.get("name")})
            except Exception as exc:
                dead.append({"ob": ob, "error": str(exc)[:120]})
    cache = {
        "refreshed_at": _now_iso(),
        "lists": [w.get("name") for w in lists],
        "entries": entries,
        "dead": dead,
    }
    atomic_write_json(CACHE_FILE, cache)
    logger.info(
        "watchlist refreshed: %d instruments, %d dead, %d list(s)",
        len(entries),
        len(dead),
        len(lists),
    )
    return cache


def load_cache():
    from portfolio.file_utils import load_json

    return load_json(CACHE_FILE, default=None)


def cache_age_seconds(cache=None):
    cache = cache or load_cache()
    if not cache or not cache.get("refreshed_at"):
        return None
    try:
        ts = datetime.datetime.fromisoformat(cache["refreshed_at"])
    except (TypeError, ValueError):
        return None
    return (datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds()


def ensure_fresh(api_get_fn=None, max_age=REFRESH_SECONDS):
    """Daily-refresh gate, called from the loop. Serves a stale cache when the
    refresh fails — a watchlist from yesterday beats no watchlist, and the age
    is visible in the snapshot."""
    cache = load_cache()
    age = cache_age_seconds(cache)
    if cache and age is not None and age < max_age:
        return cache
    try:
        return refresh_cache(api_get_fn=api_get_fn)
    except Exception as exc:
        logger.warning("watchlist refresh failed, serving stale cache: %s", exc)
        return cache


def split_entries(cache):
    """(overlap, fresh, quote_only) — overlap rows point at the Swedbank book."""
    from portfolio.swedbank.instruments import by_orderbook

    overlap, fresh, quote_only = [], [], []
    for e in (cache or {}).get("entries") or []:
        try:
            inst = by_orderbook(e["ob"])
            overlap.append({**e, "swedbank_key": inst.key})
            continue
        except KeyError:
            pass
        (quote_only if e["ob"] in QUOTE_ONLY_OBS else fresh).append(e)
    return overlap, fresh, quote_only


def _instrument_for(entry):
    ob = entry["ob"]
    name = entry["name"]
    tick = entry.get("ticker") or ""
    leveraged = ob in UNDERLYING_BY_OB
    key = _slug(tick if " " not in tick and tick else name, ob)
    return Instrument(
        key=key,
        name=name,
        asset_class=AssetClass.CERTIFICATE if leveraged else AssetClass.EQUITY,
        currency=entry.get("currency") or "SEK",
        avanza_ob=ob,
        alpaca=None,
        venue="STO" if (entry.get("currency") == "SEK") else "US",
    )


def evaluate_watchlist(cache=None, horizon="1d", evaluate_fn=None):
    """Signals + news for watchlist names not already in the book."""
    from portfolio.swedbank import signals as sigmod

    cache = cache or load_cache()
    overlap, fresh, quote_only = split_entries(cache)
    if evaluate_fn is not None:
        evaluate = evaluate_fn
    else:
        # Bind the config once. Without it signal_engine has no API keys and
        # every crypto-news fetch in the pass 401s (see signals._default_config).
        cfg = sigmod._default_config()
        evaluate = partial(sigmod.evaluate, config=cfg)

    results = {}
    for e in fresh:
        inst = _instrument_for(e)
        underlying = UNDERLYING_BY_OB.get(e["ob"])
        try:
            if underlying:
                # A leveraged product's own series is decaying noise; compute on
                # the underlying via the shared UNDERLYING mechanism instead.
                row = _evaluate_via_underlying(inst, underlying, horizon, evaluate)
            else:
                row = evaluate(inst, horizon=horizon)
        except Exception as exc:
            row = {"key": inst.key, "error": f"{type(exc).__name__}: {exc}"}
        row["ob"] = e["ob"]
        row["watch_only"] = True
        results[inst.key] = row

    for e in quote_only:
        results[_slug(e["name"], e["ob"])] = {
            "key": _slug(e["name"], e["ob"]),
            "ob": e["ob"],
            "name": e["name"],
            "watch_only": True,
            "quote_only": True,
            "note": "no signal source for this product type",
        }

    pointers = {
        e["swedbank_key"]: {
            "ob": e["ob"],
            "see": "swedbank",
            "swedbank_key": e["swedbank_key"],
        }
        for e in overlap
    }
    return results, pointers


def _evaluate_via_underlying(inst, underlying, horizon, evaluate):
    """Route a leveraged product through its underlying's series, the same way
    signals.UNDERLYING does for the pinned Tier-3 products."""
    from portfolio.swedbank import signals as sigmod

    original = sigmod.UNDERLYING.get(inst.key)
    sigmod.UNDERLYING[inst.key] = underlying
    try:
        return evaluate(inst, horizon=horizon)
    finally:
        if original is None:
            sigmod.UNDERLYING.pop(inst.key, None)
        else:
            sigmod.UNDERLYING[inst.key] = original


def write_snapshot(results, pointers, cache):
    from portfolio.file_utils import atomic_append_jsonl, atomic_write_json

    snap = {
        "as_of": _now_iso(),
        "watchlist_refreshed_at": (cache or {}).get("refreshed_at"),
        "lists": (cache or {}).get("lists"),
        "dead": (cache or {}).get("dead"),
        "tracked": results,
        "in_swedbank_book": pointers,
    }
    atomic_write_json(SNAPSHOT_FILE, snap)
    try:
        atomic_append_jsonl(
            SIGNAL_LOG,
            {
                "ts": snap["as_of"],
                "tickers": {
                    k: {
                        "action": v.get("action"),
                        "confidence": v.get("confidence"),
                        "regime": v.get("regime"),
                        "error": v.get("error"),
                    }
                    for k, v in results.items()
                    if not v.get("quote_only")
                },
            },
        )
    except Exception as exc:
        logger.warning("watchlist signal log append failed: %s", exc)
    return snap
