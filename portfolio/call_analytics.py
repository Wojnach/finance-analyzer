"""Lifetime accuracy of journalled analytical calls, broken out by dimension.

The point is not a single hit rate — that number is easy to feel good about and
tells you nothing actionable. What matters is WHERE the judgment is reliable:

* **Calibration** is the most valuable axis. If "high confidence" calls do not
  beat "low confidence" calls, the confidence signal is decoration and should be
  ignored rather than trusted. Brier score measures the stated p_up against
  outcomes; 0.25 is a coin flip, lower is better, and a score ABOVE 0.25 means
  the probabilities are worse than guessing.
* **Evidence attribution** answers which inputs actually predict. Basis strings
  are keyword-tagged, so a tag's hit rate is the hit rate of calls that CITED it,
  not proof that it caused the outcome — several tags appear on the same call.
* **Amendment effect** tracks whether revising a call helped. Revising toward a
  historical mean and away from situational signals can make a call worse, and
  that has already happened once here, so it must be measurable.

Every breakdown reports `n`. With single-digit samples these are descriptions of
what happened, not estimates of skill — `MIN_N_FOR_SIGNAL` marks which rows are
too thin to read as anything else.
"""

from __future__ import annotations

MIN_N_FOR_SIGNAL = 10

# Evidence categories, matched against the free-text basis entries a call
# carries. Deliberately coarse: the goal is "do analyst targets predict better
# than technicals", not a taxonomy.
EVIDENCE_TAGS = {
    "analyst_target": ("consensus", "price target", " pt ", "pt ", "analyst"),
    "own_history": ("earnings history", "own ", "prints", "print history", "9-for-9"),
    "insider": ("insider", "form 4", "director sold"),
    "technical": ("rsi", "ma50", "ma200", "run-in", "momentum", "overbought"),
    "options": ("implied", "straddle", "iv ", "collar", "option"),
    "monte_carlo": ("mc:", "monte carlo", "p_up", "p(-10", "bootstrap", "1m mc"),
    "macro": ("fed", "fomc", "warsh", "hike", "riksbank", "cpif", "rate"),
    "flows": ("etf", "flow", "outflow", "inflow", "short interest"),
    "valuation": ("p/e", "pe ", "forward p", "x earnings", "multiple", "234x"),
    "fundamentals": ("guidance", "revenue", "margin", "backlog", "bookings", "beat"),
    "signal_engine": ("multi-timeframe", "1h sell", "signal", "regime"),
}


def _tags_for(basis):
    text = " ".join(basis or []).lower()
    return sorted(
        t for t, keys in EVIDENCE_TAGS.items() if any(k in text for k in keys)
    )


def _rate(correct, n):
    return round(100 * correct / n, 1) if n else None


def _agg(rows, keyfn):
    """Group resolved rows by a key function -> hit-rate summary per group."""
    out = {}
    for r in rows:
        for key in keyfn(r):
            if key is None:
                continue
            a = out.setdefault(
                str(key), {"n": 0, "correct": 0, "moves": [], "abs_err": []}
            )
            a["n"] += 1
            if r.get("direction_correct") is True:
                a["correct"] += 1
            if r.get("realised_move_pct") is not None:
                a["moves"].append(r["realised_move_pct"])
                exp = r.get("_expected_move_pct")
                if exp is not None:
                    a["abs_err"].append(abs(r["realised_move_pct"] - exp))
    for a in out.values():
        a["hit_rate"] = _rate(a["correct"], a["n"])
        a["mean_move_pct"] = (
            round(sum(a["moves"]) / len(a["moves"]), 2) if a["moves"] else None
        )
        a["mean_abs_error_pct"] = (
            round(sum(a["abs_err"]) / len(a["abs_err"]), 2) if a["abs_err"] else None
        )
        a["thin_sample"] = a["n"] < MIN_N_FOR_SIGNAL
        del a["moves"], a["abs_err"]
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


def _brier(rows):
    """Mean squared error of stated p_up against the realised direction.

    0.25 = coin flip. Above 0.25 means the stated probabilities are actively
    misleading and should be inverted or discarded, not merely widened.
    """
    scored = [
        r
        for r in rows
        if r.get("_p_up") is not None and r.get("realised_move_pct") is not None
    ]
    if not scored:
        return None
    total = 0.0
    for r in scored:
        p = float(r["_p_up"]) / 100.0
        actual = 1.0 if r["realised_move_pct"] > 0 else 0.0
        total += (p - actual) ** 2
    return {"brier": round(total / len(scored), 4), "n": len(scored), "coin_flip": 0.25}


def build(path=None):
    """Join calls to resolutions and report accuracy across every dimension."""
    from portfolio import call_journal as cj

    rows = cj.load_all(path or cj.JOURNAL_PATH)
    calls = {r["id"]: r for r in rows if r.get("kind") == "call"}
    amendments = {}
    for r in rows:
        if r.get("kind") == "amendment":
            amendments.setdefault(r.get("amends_id"), []).append(r)

    resolved = []
    for r in rows:
        if r.get("kind") != "resolution":
            continue
        call = calls.get(r.get("resolves_id")) or {}
        amds = amendments.get(r.get("resolves_id")) or []
        latest = amds[-1] if amds else {}
        # An amendment's revised figures supersede the original for scoring —
        # otherwise a call is judged against a basis its author already retracted.
        exp = latest.get("revised_expected_move_pct", call.get("expected_move_pct"))
        p_up = latest.get("revised_p_up", call.get("p_up"))
        conf = latest.get("revised_confidence", call.get("confidence"))
        resolved.append(
            {
                **r,
                "_expected_move_pct": exp,
                "_p_up": p_up,
                "_confidence": conf,
                "_horizon_days": call.get("horizon_days"),
                "_basis": call.get("basis") or [],
                "_amended": bool(amds),
                "_original_expected": call.get("expected_move_pct"),
                "_retroactive": bool(call.get("retroactive")),
            }
        )

    if not resolved:
        return {"n": 0, "note": "no resolved calls yet"}

    correct = sum(1 for r in resolved if r.get("direction_correct") is True)
    sized = [r for r in resolved if r.get("within_expected") is not None]
    errs = [
        abs(r["realised_move_pct"] - r["_expected_move_pct"])
        for r in resolved
        if r.get("realised_move_pct") is not None
        and r.get("_expected_move_pct") is not None
    ]

    return {
        "overall": {
            "n": len(resolved),
            "direction_hit_rate": _rate(correct, len(resolved)),
            "size_hit_rate": _rate(
                sum(1 for r in sized if r["within_expected"]), len(sized)
            ),
            "mean_abs_error_pct": round(sum(errs) / len(errs), 2) if errs else None,
            "thin_sample": len(resolved) < MIN_N_FOR_SIGNAL,
        },
        "calibration": _brier(resolved),
        "by_confidence": _agg(resolved, lambda r: [r.get("_confidence")]),
        "by_instrument": _agg(resolved, lambda r: [r.get("instrument")]),
        "by_call_type": _agg(resolved, lambda r: [r.get("call")]),
        "by_horizon": _agg(
            resolved,
            lambda r: [
                (
                    None
                    if r.get("_horizon_days") is None
                    else ("1-7d" if r["_horizon_days"] <= 7 else "8-90d")
                )
            ],
        ),
        "by_evidence_cited": _agg(resolved, lambda r: _tags_for(r.get("_basis"))),
        "amendment_effect": _amendment_effect(resolved),
        "retroactive_count": sum(1 for r in resolved if r["_retroactive"]),
    }


def _amendment_effect(resolved):
    """Did revising a call improve its magnitude estimate, or worsen it?"""
    amended = [
        r
        for r in resolved
        if r["_amended"]
        and r.get("realised_move_pct") is not None
        and r.get("_original_expected") is not None
        and r.get("_expected_move_pct") is not None
    ]
    if not amended:
        return {"n": 0}
    better = worse = 0
    for r in amended:
        orig_err = abs(r["realised_move_pct"] - r["_original_expected"])
        new_err = abs(r["realised_move_pct"] - r["_expected_move_pct"])
        if new_err < orig_err:
            better += 1
        elif new_err > orig_err:
            worse += 1
    return {
        "n": len(amended),
        "amendment_improved": better,
        "amendment_worsened": worse,
        "note": (
            "revising toward a historical mean and away from situational signals "
            "can make a call worse; this row is the check on that"
        ),
    }


def report(path=None):
    """Human-readable lifetime scorecard."""
    a = build(path)
    if a.get("n") == 0:
        return "No resolved calls yet — nothing to score."
    L = []
    o = a["overall"]
    L.append("=== LIFETIME CALL ACCURACY ===")
    L.append(
        f"resolved {o['n']} · direction {o['direction_hit_rate']}% · "
        f"size {o['size_hit_rate']}% · mean abs error {o['mean_abs_error_pct']}pp"
        + ("   [THIN SAMPLE]" if o["thin_sample"] else "")
    )
    c = a.get("calibration")
    if c:
        verdict = (
            "better than coin flip" if c["brier"] < 0.25 else "WORSE than coin flip"
        )
        L.append(
            f"calibration: Brier {c['brier']} on n={c['n']} ({verdict}; 0.25=coin)"
        )
    ae = a.get("amendment_effect") or {}
    if ae.get("n"):
        L.append(
            f"amendments: {ae['n']} scored — improved {ae['amendment_improved']}, "
            f"worsened {ae['amendment_worsened']}"
        )
    for label, key in (
        ("BY CONFIDENCE (is our confidence meaningful?)", "by_confidence"),
        ("BY CALL TYPE", "by_call_type"),
        ("BY HORIZON", "by_horizon"),
        ("BY INSTRUMENT", "by_instrument"),
        ("BY EVIDENCE CITED (not causal — tags co-occur)", "by_evidence_cited"),
    ):
        d = a.get(key) or {}
        if not d:
            continue
        L.append(f"\n{label}")
        L.append(f"  {'group':<18}{'n':>4}{'hit%':>7}{'mean move':>11}{'abs err':>9}")
        for g, v in d.items():
            hit = "-" if v["hit_rate"] is None else f"{v['hit_rate']:.0f}"
            mm = "-" if v["mean_move_pct"] is None else f"{v['mean_move_pct']:+.1f}%"
            er = (
                "-"
                if v["mean_abs_error_pct"] is None
                else f"{v['mean_abs_error_pct']:.1f}"
            )
            flag = " *thin" if v["thin_sample"] else ""
            L.append(f"  {g[:17]:<18}{v['n']:>4}{hit:>7}{mm:>11}{er:>9}{flag}")
    L.append(
        "\n* thin = fewer than %d samples; descriptive, not skill." % MIN_N_FOR_SIGNAL
    )
    return "\n".join(L)
