"""5-min trigger buffer — collapses noise into single fire-events.

Sliding window keyed on (ticker, reason_type). Within the window:
- identical reasons deduped
- distinct reasons for same (ticker, reason_type) concatenated
- escalation reason_types (T3-trigger-class: first_of_day, periodic_review,
  F&G_extreme, post_trade) bypass buffering and force-flush

Item 8 of docs/PLAN.md. State file: data/trigger_buffer.json.
Added 2026-05-15 — not yet wired into main.py (Batch D).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from portfolio.file_utils import atomic_write_json, load_json

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_STATE_FILE = BASE_DIR / "data" / "trigger_buffer.json"

ESCALATION_TYPES = {"first_of_day", "periodic_review", "F&G_extreme", "post_trade"}

# Reserved words that look like tickers but aren't.
_TICKER_BLOCKLIST = {"BUY", "SELL", "HOLD", "USD", "EUR", "SEK"}
_TICKER_RE = re.compile(r"\b([A-Z]{2,5}(?:-[A-Z]{3})?)\b")


@dataclass
class BufferedReason:
    ts: float
    ticker: str
    reason_type: str
    reason: str


def _resolve_path(state_path: Optional[str | Path]) -> Path:
    if state_path is None:
        return DEFAULT_STATE_FILE
    return Path(state_path)


def _extract_ticker(reason: str) -> str:
    for m in _TICKER_RE.finditer(reason):
        tok = m.group(1)
        if "-" in tok:
            # e.g. BTC-USD, XAU-USD — accept
            return tok
        if tok in _TICKER_BLOCKLIST:
            continue
        # Plain ticker like MSTR
        return tok
    return ""


def parse_reason(reason: str) -> tuple[str, str]:
    """Return (ticker, reason_type) for a reason string."""
    if not reason:
        return ("", "other")

    r = reason.strip()
    low = r.lower()

    # Exact-match escalation / lifecycle reasons.
    if low == "startup":
        return ("", "startup")
    if low == "first_of_day":
        return ("", "first_of_day")
    if low == "periodic_review":
        return ("", "periodic_review")

    # F&G extreme — phrases like "F&G crossed below 20" or "F&G_extreme"
    if low.startswith("f&g") or "f&g_extreme" in low:
        return ("", "F&G_extreme")

    # post-trade
    if low.startswith("post-trade") or low.startswith("post_trade"):
        return (_extract_ticker(r), "post_trade")

    ticker = _extract_ticker(r)

    # Order matters: "sentiment" before "flipped" (sentiment lines also contain "->")
    if "sentiment" in low:
        return (ticker, "sentiment")
    if "consensus" in low:
        return (ticker, "consensus")
    if "flipped" in low or "->" in r:
        return (ticker, "flipped")
    if "moved" in low:
        return (ticker, "moved")

    return (ticker, "other")


def _load_entries(path: Path) -> list[BufferedReason]:
    raw = load_json(path, default={"entries": []}) or {"entries": []}
    out: list[BufferedReason] = []
    for e in raw.get("entries", []):
        try:
            out.append(
                BufferedReason(
                    ts=float(e["ts"]),
                    ticker=str(e.get("ticker", "")),
                    reason_type=str(e.get("reason_type", "other")),
                    reason=str(e.get("reason", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _save_entries(path: Path, entries: list[BufferedReason]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"entries": [asdict(e) for e in entries]})


def add(
    reasons: list[str],
    now_ts: float,
    state_path: Optional[str | Path] = None,
) -> None:
    """Append reasons to buffer. Atomic write."""
    if not reasons:
        return
    path = _resolve_path(state_path)
    entries = _load_entries(path)
    for r in reasons:
        if not r:
            continue
        ticker, rtype = parse_reason(r)
        entries.append(
            BufferedReason(ts=float(now_ts), ticker=ticker, reason_type=rtype, reason=r)
        )
    _save_entries(path, entries)


def _merge_bucket(bucket: list[BufferedReason]) -> str:
    """Merge a (ticker, reason_type) bucket into a single representative reason."""
    # Dedupe identical reason strings, count occurrences.
    counts: dict[str, int] = {}
    order: list[str] = []
    for e in bucket:
        if e.reason not in counts:
            counts[e.reason] = 0
            order.append(e.reason)
        counts[e.reason] += 1

    total = sum(counts.values())
    # Representative = shortest reason string.
    rep = min(order, key=len)
    if total > 1:
        return f"{rep} (x{total})"
    return rep


def flush_due(
    now_ts: float,
    window_s: int = 300,
    state_path: Optional[str | Path] = None,
) -> list[str]:
    """Return reasons whose window has expired OR escalation present. Remove flushed atomically."""
    path = _resolve_path(state_path)
    entries = _load_entries(path)
    if not entries:
        return []

    has_escalation = any(e.reason_type in ESCALATION_TYPES for e in entries)

    if has_escalation:
        to_flush = entries
        remaining: list[BufferedReason] = []
    else:
        to_flush = [e for e in entries if (now_ts - e.ts) >= window_s]
        remaining = [e for e in entries if (now_ts - e.ts) < window_s]

    if not to_flush:
        return []

    # Group by (ticker, reason_type), preserve first-seen order.
    buckets: dict[tuple[str, str], list[BufferedReason]] = {}
    order: list[tuple[str, str]] = []
    for e in to_flush:
        key = (e.ticker, e.reason_type)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(e)

    merged = [_merge_bucket(buckets[k]) for k in order]

    _save_entries(path, remaining)
    return merged


def buffer_size(state_path: Optional[str | Path] = None) -> int:
    """Diagnostic — total entries in buffer."""
    path = _resolve_path(state_path)
    return len(_load_entries(path))
