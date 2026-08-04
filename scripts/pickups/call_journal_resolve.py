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


def _price_map(repo_root: Path, keys):
    """Live marks, Swedbank instruments first, watchlist orderbooks second."""
    from portfolio.fx_rates import fetch_usd_sek
    from portfolio.swedbank.instruments import INSTRUMENTS
    from portfolio.swedbank.pricing import sweep

    out = {}
    pinned = [k for k in keys if k in INSTRUMENTS]
    if pinned:
        s = sweep(keys=pinned, fx_fn=fetch_usd_sek)
        quotes = s.quotes if hasattr(s, "quotes") else s
        for k, q in quotes.items():
            if q is not None and getattr(q, "mark", None):
                out[k] = float(q.mark)

    missing = [k for k in keys if k not in out]
    if missing:
        from portfolio.avanza_session import api_get
        from portfolio.file_utils import load_json

        cache = load_json(str(repo_root / "data" / "watchlist_instruments.json")) or {}
        by_ticker = {}
        for e in cache.get("entries") or []:
            t = (e.get("ticker") or "").strip()
            if t:
                by_ticker[t] = e["ob"]
        for k in missing:
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
        return {
            "verdict": "reviewed",
            "summary": (
                f"Resolved {len(resolved)}/{len(due)} due calls ({wins} correct). "
                f"{lines}. Running scorecard: n={sc['n']}, "
                f"direction hit rate={sc.get('direction_hit_rate')}%."
                + (f" Unpriced, left open: {skipped}." if skipped else "")
            ),
            "resolved": len(resolved),
            "skipped": skipped,
            "scorecard": sc,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        return {
            "verdict": "error",
            "summary": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }
