"""Pickup handler: score any analytical calls whose horizon has elapsed.

Reads `data/call_journal.jsonl`, finds open calls past their `resolve_after`,
prices each instrument live, and appends a resolution line per call. Then reports
the running scorecard so a future session sees whether these calls are actually
any good — which is the only way the judgment layer ever gets calibrated.

Never raises: a pricing outage must leave the calls open for the next run rather
than banking a resolution against a stale or missing price.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _swedbank_prices(keys):
    """Marks for instruments pinned in the Swedbank book."""
    from portfolio.fx_rates import fetch_usd_sek
    from portfolio.swedbank.instruments import INSTRUMENTS
    from portfolio.swedbank.pricing import sweep

    out = {}
    pinned = [k for k in keys if k in INSTRUMENTS]
    if not pinned:
        return out
    s = sweep(keys=pinned, fx_fn=fetch_usd_sek)
    quotes = s.quotes if hasattr(s, "quotes") else s
    for k, q in quotes.items():
        if q is not None and getattr(q, "mark", None):
            out[k] = float(q.mark)
    return out


def _watchlist_prices(repo_root: Path, keys):
    """Last prints for tickers carrying an Avanza watchlist orderbook ID."""
    from portfolio.avanza_session import api_get
    from portfolio.file_utils import load_json

    out = {}
    cache = load_json(str(repo_root / "data" / "watchlist_instruments.json")) or {}
    by_ticker = {}
    for e in cache.get("entries") or []:
        t = (e.get("ticker") or "").strip()
        if t:
            by_ticker[t] = e["ob"]
    for k in keys:
        ob = by_ticker.get(k)
        if not ob:
            continue
        try:
            g = api_get(f"/_api/market-guide/stock/{ob}")
            last = (g.get("quote") or {}).get("last")
            if last:
                out[k] = float(last)
        except Exception:
            continue
    return out


def _price_source_prices(keys):
    """Last close from portfolio.price_source — the Tier-1 universe.

    Added 2026-08-17. Without this, XAU-USD/XAG-USD could never be priced:
    they are Binance FAPI synthetics, absent from both the Swedbank book and
    the Avanza watchlist, so CALLS-VERIFY-1D reported "Resolved 0/2 due calls,
    unpriced: ['XAG-USD', 'XAU-USD']" and would have deferred forever — the
    scoring pickup silently failing to score is the one outcome that defeats
    its whole purpose.
    """
    out = {}
    for k in keys:
        try:
            from portfolio.price_source import fetch_klines

            df = fetch_klines(k, interval="1h", limit=2)
            close = df["close"]
            last = close.iloc[-1] if hasattr(close, "iloc") else close[-1]
            if last:
                out[k] = float(last)
        except Exception:
            continue
    return out


def _price_map(repo_root: Path, keys):
    """Live marks in source order, most authoritative first.

    Avanza-backed sources win where they have a quote — they are the venue the
    operator actually trades — and price_source backfills everything else.
    Never raises: a pricing outage must leave calls open for the next run
    rather than bank a resolution against a missing price.
    """
    out = {}
    for tier in (
        lambda ks: _swedbank_prices(ks),
        lambda ks: _watchlist_prices(repo_root, ks),
        lambda ks: _price_source_prices(ks),
    ):
        missing = [k for k in keys if k not in out]
        if not missing:
            break
        try:
            for k, v in (tier(missing) or {}).items():
                if k not in out:
                    out[k] = v
        except Exception:
            continue
    return out


def run(pickup: dict, repo_root: Path) -> dict:
    try:
        sys.path.insert(0, str(repo_root))
        from portfolio import call_journal as cj

        path = str(repo_root / "data" / "call_journal.jsonl")
        opens = cj.open_calls(path=path)
        due = [c for c in opens if c.get("due")]
        if not due:
            sc = cj.scorecard(path=path)
            return {
                "verdict": "defer",
                "summary": (
                    f"No calls due yet ({len(opens)} open). "
                    f"Scorecard so far: n={sc.get('n', 0)}, "
                    f"direction hit rate={sc.get('direction_hit_rate')}%."
                ),
                "open": len(opens),
            }

        prices = _price_map(repo_root, sorted({c["instrument"] for c in due}))
        resolved, skipped = [], []
        for c in due:
            p = prices.get(c["instrument"])
            if p is None:
                skipped.append(c["instrument"])
                continue
            r = cj.resolve_call(c, p, path=path, note="auto-resolved by pickup")
            resolved.append(r)

        sc = cj.scorecard(path=path)
        wins = sum(1 for r in resolved if r["verdict"] == "correct")
        lines = "; ".join(
            f"{r['instrument']} {r['call']} -> {r['verdict']} ({r['realised_move_pct']:+.1f}%)"
            for r in resolved
        )

        # Lifetime breakdown, not just the running total — a single hit rate is
        # the one number that cannot tell you where the judgment is unreliable.
        analytics, report_text = None, ""
        try:
            from portfolio import call_analytics as ca

            analytics = ca.build(path=path)
            report_text = ca.report(path=path)
        except Exception as exc:  # noqa: BLE001
            report_text = f"(analytics unavailable: {type(exc).__name__}: {exc})"

        cal = (analytics or {}).get("calibration") or {}
        cal_note = (
            f" Brier {cal['brier']} vs 0.25 coin-flip on n={cal['n']}."
            if cal.get("brier") is not None
            else ""
        )
        return {
            "verdict": "reviewed",
            "summary": (
                f"Resolved {len(resolved)}/{len(due)} due calls ({wins} correct). "
                f"{lines}. Running scorecard: n={sc['n']}, "
                f"direction hit rate={sc.get('direction_hit_rate')}%.{cal_note}"
                + (f" Unpriced, left open: {skipped}." if skipped else "")
            ),
            "resolved": len(resolved),
            "skipped": skipped,
            "scorecard": sc,
            "analytics": analytics,
            "details": {"lifetime_report": report_text},
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        return {
            "verdict": "error",
            "summary": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }
