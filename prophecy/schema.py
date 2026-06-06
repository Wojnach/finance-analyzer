"""Prophecy prediction record schema + validation.

Pure-Python (no pydantic dependency). One *record* = one instrument's full
forecast for a single daily run: a direction/target/probability/confidence
prediction at each of the 10 fixed horizons, plus a ``coverage`` block that
flags where the system lacks the data or equations to predict credibly.

Design rule (premortem #4): ``validate_record`` may *repair* soft issues
(normalise probabilities, clamp confidence) but NEVER upgrades
``coverage.needs_work`` from True to False — only the producer (prep seeds it,
Claude may downgrade) is allowed to claim "we have enough".
"""

from __future__ import annotations

from datetime import timedelta

# --- horizons -------------------------------------------------------------
# 1d == "today" (next session close). Order matters: dashboard renders L->R.
HORIZONS: list[str] = ["1d", "2d", "3d", "4d", "5d", "6d", "7d", "1mo", "2mo", "6mo"]

# How far ahead each horizon resolves, for outcome backfill. Calendar-day
# approximations (markets/crypto differ; outcomes.py tolerates a window).
HORIZON_DELTAS: dict[str, timedelta] = {
    "1d": timedelta(days=1),
    "2d": timedelta(days=2),
    "3d": timedelta(days=3),
    "4d": timedelta(days=4),
    "5d": timedelta(days=5),
    "6d": timedelta(days=6),
    "7d": timedelta(days=7),
    "1mo": timedelta(days=30),
    "2mo": timedelta(days=60),
    "6mo": timedelta(days=180),
}

DIRECTIONS = {"up", "down", "flat"}
SUFFICIENCY_LEVELS = ["high", "medium", "low", "insufficient"]
# A run with sufficiency in this set, or no proper equation, needs engineering work.
_NEEDS_WORK_SUFFICIENCY = {"low", "insufficient"}


# --- coverage / gap flag --------------------------------------------------
def grade_sufficiency(found: int, required: int) -> str:
    """Grade data sufficiency from #inputs-found vs #inputs-required.

    Deterministic seed used by prep.py before Claude ever runs, so the gap
    flag is meaningful even if the Claude step is skipped/fails.
    """
    if required <= 0:
        return "insufficient"
    ratio = found / required
    if ratio >= 0.85:
        return "high"
    if ratio >= 0.6:
        return "medium"
    if ratio >= 0.3:
        return "low"
    return "insufficient"


def build_coverage(
    *,
    data_sufficiency: str,
    has_proper_equation: bool,
    missing_inputs: list[str] | None = None,
    low_confidence_horizons: list[str] | None = None,
    note: str = "",
) -> dict:
    """Construct a coverage block, deriving ``needs_work`` consistently."""
    if data_sufficiency not in SUFFICIENCY_LEVELS:
        data_sufficiency = "insufficient"
    needs_work = (data_sufficiency in _NEEDS_WORK_SUFFICIENCY) or (not has_proper_equation)
    return {
        "data_sufficiency": data_sufficiency,
        "has_proper_equation": bool(has_proper_equation),
        "missing_inputs": list(missing_inputs or []),
        "low_confidence_horizons": list(low_confidence_horizons or []),
        "needs_work": bool(needs_work),
        "note": str(note or ""),
    }


def _coerce_coverage(raw: dict | None) -> dict:
    """Normalise an arbitrary coverage dict; default to needs_work=True.

    A missing/garbage coverage block is treated as a gap (conservative):
    absence of evidence that we have enough data == needs work.
    """
    raw = raw if isinstance(raw, dict) else {}
    suff = raw.get("data_sufficiency")
    if suff not in SUFFICIENCY_LEVELS:
        suff = "insufficient"
    has_eq = bool(raw.get("has_proper_equation", False))
    cov = build_coverage(
        data_sufficiency=suff,
        has_proper_equation=has_eq,
        missing_inputs=raw.get("missing_inputs") or [],
        low_confidence_horizons=raw.get("low_confidence_horizons") or [],
        note=raw.get("note", ""),
    )
    # NEVER allow a producer-claimed needs_work=True to be silently cleared.
    if raw.get("needs_work") is True:
        cov["needs_work"] = True
    return cov


# --- per-horizon prediction ----------------------------------------------
def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _validate_horizon(name: str, raw: dict, errors: list[str]) -> dict | None:
    """Validate+repair one horizon prediction. Returns None if unsalvageable."""
    if not isinstance(raw, dict):
        errors.append(f"{name}: not an object")
        return None

    direction = str(raw.get("direction", "")).lower().strip()
    if direction not in DIRECTIONS:
        errors.append(f"{name}: bad direction {raw.get('direction')!r}")
        return None

    try:
        target = float(raw.get("target"))
    except (TypeError, ValueError):
        errors.append(f"{name}: target not numeric ({raw.get('target')!r})")
        return None
    if not (target > 0):
        errors.append(f"{name}: target <= 0 ({target})")
        return None

    # Probabilities: normalise to sum 1 (soft repair). Default to direction.
    p_up = _safe_float(raw.get("prob_up"))
    p_down = _safe_float(raw.get("prob_down"))
    p_flat = _safe_float(raw.get("prob_flat"))
    if p_up is None and p_down is None and p_flat is None:
        # No probs given: synthesise a weak prior from direction.
        p_up, p_down, p_flat = (
            (0.55, 0.30, 0.15) if direction == "up"
            else (0.30, 0.55, 0.15) if direction == "down"
            else (0.30, 0.30, 0.40)
        )
        errors.append(f"{name}: no probabilities, synthesised from direction")
    else:
        p_up, p_down, p_flat = (p_up or 0.0), (p_down or 0.0), (p_flat or 0.0)
    total = p_up + p_down + p_flat
    if total <= 0:
        errors.append(f"{name}: probabilities sum to 0")
        return None
    p_up, p_down, p_flat = p_up / total, p_down / total, p_flat / total

    confidence = _safe_float(raw.get("confidence"))
    confidence = 0.5 if confidence is None else _clamp(confidence, 0.0, 1.0)

    low = _safe_float(raw.get("low"))
    high = _safe_float(raw.get("high"))
    if low is not None and high is not None and low > high:
        low, high = high, low  # swap obvious inversion

    return {
        "direction": direction,
        "target": round(target, 6),
        "prob_up": round(p_up, 4),
        "prob_down": round(p_down, 4),
        "prob_flat": round(p_flat, 4),
        "confidence": round(confidence, 4),
        "low": round(low, 6) if low is not None else None,
        "high": round(high, 6) if high is not None else None,
        "rationale": str(raw.get("rationale", ""))[:2000],
    }


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    # Reject NaN/inf (would poison JSON + downstream math).
    return out if out == out and abs(out) != float("inf") else None


def validate_record(raw: dict) -> tuple[dict | None, list[str]]:
    """Validate+repair one instrument record.

    Returns ``(clean_record, errors)``. ``clean_record`` is None only when the
    record is unusable (no instrument, or zero valid horizons). A record with
    *some* valid horizons is kept; missing horizons are listed in errors and
    pushed into coverage.low_confidence_horizons + needs_work.
    """
    errors: list[str] = []
    if not isinstance(raw, dict):
        return None, ["record: not an object"]

    instrument = str(raw.get("instrument", "")).strip()
    if not instrument:
        return None, ["record: missing instrument"]

    horizons_in = raw.get("horizons")
    if not isinstance(horizons_in, dict):
        return None, [f"{instrument}: missing/invalid horizons block"]

    clean_horizons: dict[str, dict] = {}
    missing: list[str] = []
    for h in HORIZONS:
        validated = _validate_horizon(h, horizons_in.get(h, {}), errors) if h in horizons_in else None
        if validated is None:
            missing.append(h)
        else:
            clean_horizons[h] = validated

    if not clean_horizons:
        return None, errors + [f"{instrument}: zero valid horizons"]

    coverage = _coerce_coverage(raw.get("coverage"))
    if missing:
        errors.append(f"{instrument}: missing horizons {missing}")
        # Missing horizons are themselves a gap.
        lcw = set(coverage["low_confidence_horizons"]) | set(missing)
        coverage["low_confidence_horizons"] = sorted(lcw, key=lambda x: HORIZONS.index(x) if x in HORIZONS else 99)
        coverage["needs_work"] = True

    clean = {
        "schema_version": 1,
        "instrument": instrument,
        "strategy": str(raw.get("strategy", "")),
        "model": str(raw.get("model", "")),
        "regime": str(raw.get("regime", "")),
        "horizons": clean_horizons,
        "key_drivers": _str_list(raw.get("key_drivers"), 30),
        "stored_signals_used": _str_list(raw.get("stored_signals_used"), 80),
        "web_sources": _str_list(raw.get("web_sources"), 40),
        "forum_sentiment": raw.get("forum_sentiment") if isinstance(raw.get("forum_sentiment"), dict) else {},
        "deep_research_summary": str(raw.get("deep_research_summary", ""))[:8000],
        "coverage": coverage,
        # spot_at_prediction / ts / run_id / cost are stamped by publish.py.
    }
    return clean, errors


def _str_list(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x)[:500] for x in value[:limit]]
