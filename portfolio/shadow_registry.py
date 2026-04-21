"""Shadow-signal registry with promotion criteria and age tracking.

Purpose
-------
A signal enters "shadow" mode when its output is logged but not voted —
typically because we want to accumulate ground-truth data before trusting
its votes. Without explicit tracking, signals get forgotten in shadow for
months (FinGPT sat ~3 weeks without a single accuracy measurement, Kronos
ran 3668 predictions in shadow mode that all collapsed to HOLD). This
registry records:

* When each signal entered shadow.
* What promotion criteria were agreed.
* When it was last reviewed.
* A resolution outcome when it exits (promoted / retired / still-shadow).

The registry is a plain JSON file — no DB dependency. `scripts/review_shadow_signals.py`
reads it and flags any shadow older than 30 days without a resolution.

Schema
------
```json
{
  "shadows": {
    "fingpt": {
      "entered_shadow_ts": "2026-04-09T00:00:00+00:00",
      "promotion_criteria": {
        "min_samples": 200,
        "min_accuracy": 0.60,
        "max_missing_outcome_rate": 0.20
      },
      "last_reviewed_ts": "2026-04-21T13:45:00+00:00",
      "status": "shadow",
      "notes": "Parser fix shipped 2026-04-09; outcome backfill pending."
    }
  }
}
```

`status` is one of: `"shadow"`, `"promoted"`, `"retired"`.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.shadow_registry")

_BASE_DIR = Path(__file__).resolve().parent.parent
_REGISTRY_FILE = _BASE_DIR / "data" / "shadow_registry.json"
_STALE_DAYS = 30

_VALID_STATUS = frozenset({"shadow", "promoted", "retired"})


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def load_registry(path: Path | str | None = None) -> dict:
    """Load the registry. Returns `{"shadows": {}}` when the file is
    missing or malformed (never raises)."""
    p = Path(path) if path else _REGISTRY_FILE
    data = load_json(str(p), default=None)
    if not isinstance(data, dict) or "shadows" not in data:
        return {"shadows": {}}
    return data


def save_registry(data: dict, path: Path | str | None = None) -> None:
    """Atomically write the registry."""
    p = Path(path) if path else _REGISTRY_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(p), data)


def add_shadow(
    signal: str,
    promotion_criteria: dict,
    notes: str = "",
    *,
    entered_ts: str | None = None,
    path: Path | str | None = None,
) -> None:
    """Register `signal` as entering shadow. If already present, update
    `promotion_criteria` and `notes`, reset status to `"shadow"`, and
    refresh `last_reviewed_ts` — but PRESERVE `entered_shadow_ts` so
    days-in-shadow accounting survives re-registration."""
    reg = load_registry(path=path)
    now = _now()
    existing = reg["shadows"].get(signal, {})
    entered = entered_ts or existing.get("entered_shadow_ts") or now
    reg["shadows"][signal] = {
        "entered_shadow_ts": entered,
        "promotion_criteria": dict(promotion_criteria),
        "last_reviewed_ts": now,
        "status": "shadow",
        "notes": notes or existing.get("notes", ""),
    }
    save_registry(reg, path=path)


def resolve_shadow(
    signal: str,
    status: str,
    notes: str = "",
    *,
    path: Path | str | None = None,
) -> bool:
    """Mark a shadow as promoted or retired. Returns True if found, False
    otherwise. Does NOT delete the entry — keeps resolution history."""
    if status not in _VALID_STATUS:
        raise ValueError(f"status must be in {_VALID_STATUS}, got {status!r}")
    reg = load_registry(path=path)
    entry = reg["shadows"].get(signal)
    if entry is None:
        return False
    entry["status"] = status
    entry["last_reviewed_ts"] = _now()
    if notes:
        entry["notes"] = notes
    save_registry(reg, path=path)
    return True


def days_in_shadow(signal: str, *, path: Path | str | None = None,
                    now: _dt.datetime | None = None) -> float | None:
    """Return days elapsed since signal entered shadow. None if unknown."""
    reg = load_registry(path=path)
    entry = reg["shadows"].get(signal)
    if entry is None:
        return None
    entered_raw = entry.get("entered_shadow_ts")
    if not entered_raw:
        return None
    try:
        entered = _dt.datetime.fromisoformat(entered_raw)
    except (TypeError, ValueError):
        return None
    if entered.tzinfo is None:
        entered = entered.replace(tzinfo=_dt.UTC)
    cur = now or _dt.datetime.now(_dt.UTC)
    return (cur - entered).total_seconds() / 86400.0


def stale_shadows(*, stale_days: int = _STALE_DAYS,
                   path: Path | str | None = None,
                   now: _dt.datetime | None = None) -> list[dict]:
    """Return shadow entries that are still `"shadow"` and older than
    `stale_days`. Each dict includes `signal`, `days_in_shadow`, and the
    full entry for convenience."""
    reg = load_registry(path=path)
    stale = []
    for sig, entry in reg["shadows"].items():
        if entry.get("status") != "shadow":
            continue
        age = days_in_shadow(sig, path=path, now=now)
        if age is None:
            continue
        if age >= stale_days:
            stale.append({
                "signal": sig,
                "days_in_shadow": age,
                **entry,
            })
    return sorted(stale, key=lambda x: -x["days_in_shadow"])


def seed_defaults(path: Path | str | None = None) -> None:
    """Idempotent one-time seeding for the 2026-04-21 LLM-health audit.
    Only adds entries that don't already exist — safe to re-run."""
    reg = load_registry(path=path)
    defaults = {
        "fingpt": {
            "entered_shadow_ts": "2026-04-09T00:00:00+00:00",
            "promotion_criteria": {
                "min_samples": 200,
                "min_accuracy": 0.60,
                "max_missing_outcome_rate": 0.20,
            },
            "notes": "Parser fix shipped 2026-04-09 (fde9cf8+28aa5d0). "
                     "Accuracy vs outcomes not yet measured — awaiting "
                     "outcome backfill for sentiment_ab_log.",
        },
        "finbert": {
            "entered_shadow_ts": "2026-04-09T00:00:00+00:00",
            "promotion_criteria": {
                "min_samples": 200,
                "min_accuracy": 0.60,
                "max_missing_outcome_rate": 0.20,
            },
            "notes": "CPU-cheap shadow alongside CryptoBERT/Trading-Hero-LLM. "
                     "86% neutral output, 87.9% primary-agreement — likely "
                     "collapsed to safe-label. Keep for observation only.",
        },
        "kronos": {
            "entered_shadow_ts": "2026-03-27T15:10:00+00:00",
            "promotion_criteria": {
                "min_samples": 500,
                "min_accuracy": 0.55,
                "min_subprocess_success_rate": 0.90,
            },
            "notes": "Un-retired 2026-04-21 afternoon with proper vote-pool "
                     "isolation (shadow sub-signal excluded from "
                     "_health_weighted_vote). Subprocess reliability still "
                     "59% — fix is a separate work stream.",
        },
        "credit_spread_risk": {
            "entered_shadow_ts": "2026-04-11T00:00:00+00:00",
            "promotion_criteria": {
                "min_samples": 200,
                "min_accuracy": 0.55,
            },
            "notes": "Registered but force-HOLD via DISABLED_SIGNALS pending "
                     "live validation.",
        },
        "crypto_macro": {
            "entered_shadow_ts": "2026-04-11T00:00:00+00:00",
            "promotion_criteria": {
                "min_samples": 200,
                "min_accuracy": 0.55,
            },
            "notes": "Registered but force-HOLD via DISABLED_SIGNALS pending "
                     "live validation.",
        },
    }
    for sig, cfg in defaults.items():
        if sig in reg["shadows"]:
            continue
        reg["shadows"][sig] = {
            **cfg,
            "last_reviewed_ts": _now(),
            "status": "shadow",
        }
    save_registry(reg, path=path)
