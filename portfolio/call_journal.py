"""Append-only journal of dated analytical calls, so they can be scored later.

Distinct from the signal logs: `signal_log.jsonl` records what the engine voted,
this records what an ANALYST (human or agent) concluded, with the evidence it
rested on and a date after which it can be judged. Without it, every session
re-derives an opinion and nobody ever learns whether the last one was right.

Append-only, following the `critical_errors.jsonl` convention: a resolution is a
NEW line carrying `resolves_id`, never an edit of the original. A journal you can
rewrite is a journal that will quietly agree with whatever happened.

Scoring is deliberately mechanical. `direction_correct` compares the realised
move against the call, and `within_expected` checks whether the move landed
inside the stated range — so a call that was directionally right for the wrong
magnitude is visible as such rather than banked as a win.
"""

from __future__ import annotations

import datetime
import logging

logger = logging.getLogger("portfolio.call_journal")

JOURNAL_PATH = "data/call_journal.jsonl"

CALLS = {"BUY", "SELL", "TRIM", "HOLD", "AVOID"}
# A call needs a direction to be scorable. HOLD/AVOID express "do nothing" and
# are scored on whether the avoided move would have hurt.
_BEARISH = {"SELL", "TRIM", "AVOID"}


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def make_id(instrument, ts=None):
    ts = ts or _now()
    # Microseconds matter: a second-resolution id collides when two calls on the
    # same instrument are logged in the same second, and a duplicate id makes
    # resolutions attach to the wrong call — silently overwriting its basis and
    # corrupting every per-evidence and per-confidence breakdown built from it.
    return f"{instrument}-{ts.strftime('%Y%m%dT%H%M%S')}-{ts.microsecond:06d}"


def log_call(
    instrument,
    call,
    thesis,
    price_at_call,
    horizon_days,
    basis=None,
    expected_move_pct=None,
    downside_pct=None,
    p_up=None,
    stakes_sek=None,
    confidence=None,
    source=None,
    path=JOURNAL_PATH,
    ts=None,
    retroactive=False,
):
    """Record one call. Returns the entry (including its generated id).

    `retroactive=True` logs a call that was made verbally but never journalled at
    the time. It must be flagged, because a record containing only the calls its
    author felt good about will report a hit rate that means nothing.
    """
    from portfolio.file_utils import atomic_append_jsonl

    call = (call or "").upper()
    if call not in CALLS:
        raise ValueError(f"call must be one of {sorted(CALLS)}, got {call!r}")
    ts = ts or _now()
    resolve_after = ts + datetime.timedelta(days=max(1, int(horizon_days)))
    entry = {
        "kind": "call",
        "id": make_id(instrument, ts),
        "ts": ts.isoformat(),
        "instrument": instrument,
        "call": call,
        "thesis": thesis,
        "basis": list(basis or []),
        "price_at_call": None if price_at_call is None else float(price_at_call),
        "horizon_days": int(horizon_days),
        "resolve_after": resolve_after.isoformat(),
        "expected_move_pct": expected_move_pct,
        "downside_pct": downside_pct,
        "p_up": p_up,
        "stakes_sek": stakes_sek,
        "confidence": confidence,
        "source": source,
        "status": "open",
        "retroactive": bool(retroactive),
    }
    atomic_append_jsonl(path, entry)
    logger.info("call logged: %s %s @ %s", call, instrument, price_at_call)
    return entry


def resolve_call(call_entry, price_now, note=None, path=JOURNAL_PATH, ts=None):
    """Append a resolution for one open call. Never mutates the original line."""
    from portfolio.file_utils import atomic_append_jsonl

    ts = ts or _now()
    p0 = call_entry.get("price_at_call")
    move = None if not p0 else (float(price_now) / float(p0) - 1) * 100

    verdict, direction_correct, within = "unscorable", None, None
    if move is not None:
        bearish = call_entry["call"] in _BEARISH
        direction_correct = (move < 0) if bearish else (move > 0)
        exp = call_entry.get("expected_move_pct")
        if exp is not None:
            # "Within expected" means the realised move did not overshoot the
            # stated expectation by more than half its own size — a loose band on
            # purpose, since these are judgment calls, not point forecasts.
            band = abs(float(exp)) * 0.5 + 2.0
            within = abs(move - float(exp)) <= band
        verdict = "correct" if direction_correct else "wrong"
        if direction_correct and within is False:
            verdict = "right-direction-wrong-size"

    entry = {
        "kind": "resolution",
        "ts": ts.isoformat(),
        "resolves_id": call_entry["id"],
        "instrument": call_entry["instrument"],
        "call": call_entry["call"],
        "price_at_call": p0,
        "price_at_resolve": float(price_now),
        "realised_move_pct": None if move is None else round(move, 2),
        "direction_correct": direction_correct,
        "within_expected": within,
        "verdict": verdict,
        "note": note,
    }
    atomic_append_jsonl(path, entry)
    logger.info(
        "call resolved: %s %s -> %s (%s)",
        call_entry["call"],
        call_entry["instrument"],
        verdict,
        entry["realised_move_pct"],
    )
    return entry


def load_all(path=JOURNAL_PATH):
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def open_calls(path=JOURNAL_PATH, now=None):
    """Calls with no resolution yet. `due` marks those past resolve_after."""
    rows = load_all(path)
    resolved = {r.get("resolves_id") for r in rows if r.get("kind") == "resolution"}
    now = now or _now()
    out = []
    for r in rows:
        if r.get("kind") != "call" or r["id"] in resolved:
            continue
        try:
            due = datetime.datetime.fromisoformat(r["resolve_after"]) <= now
        except (TypeError, ValueError, KeyError):
            due = False
        out.append({**r, "due": due})
    return out


def scorecard(path=JOURNAL_PATH):
    """Hit rate over resolved calls. The whole point of the journal."""
    rows = [r for r in load_all(path) if r.get("kind") == "resolution"]
    if not rows:
        return {"n": 0}
    correct = sum(1 for r in rows if r.get("direction_correct") is True)
    sized = [r for r in rows if r.get("within_expected") is not None]
    moves = [
        r["realised_move_pct"] for r in rows if r.get("realised_move_pct") is not None
    ]
    return {
        "n": len(rows),
        "direction_hit_rate": round(100 * correct / len(rows), 1),
        "size_hit_rate": (
            round(
                100 * sum(1 for r in sized if r["within_expected"]) / len(sized),
                1,
            )
            if sized
            else None
        ),
        "mean_realised_move_pct": round(sum(moves) / len(moves), 2) if moves else None,
        "by_instrument": _by_instrument(rows),
    }


def _by_instrument(rows):
    agg = {}
    for r in rows:
        a = agg.setdefault(r["instrument"], {"n": 0, "correct": 0})
        a["n"] += 1
        if r.get("direction_correct") is True:
            a["correct"] += 1
    return {
        k: {**v, "hit_rate": round(100 * v["correct"] / v["n"], 1)}
        for k, v in agg.items()
    }
