"""Autonomous-first routing — Batch D / item 6 of docs/PLAN.md.

When ``claude_budget.autonomous_first_enabled`` is True the main loop
defaults every trigger to ``autonomous_decision`` and only escalates to
the Claude subprocess (``invoke_agent``) when one of five criteria
match:

  1. tier == 3 (F&G extreme, first-of-day, periodic 4h)
  2. Drawdown delta vs last snapshot > escalate_drawdown_pct
  3. Top-5 reliable signals (by 1d global accuracy) split BUY/SELL on
     the triggered ticker — at least 2 BUY AND at least 2 SELL
  4. Held position + SELL-side flip toward exit
  5. Post-trade trigger (reason contains "post-trade")

State (last drawdown snapshot) persists in
``data/escalation_router_state.json`` via atomic write.

Master switch defaults False — opt-in only.
"""

from __future__ import annotations

import logging
import pathlib
import re
from typing import Iterable

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DEFAULT_STATE_PATH = str(DATA_DIR / "escalation_router_state.json")
DEFAULT_ACCURACY_CACHE_PATH = str(DATA_DIR / "accuracy_cache.json")


def _norm_action(val) -> str:
    if not isinstance(val, str):
        return ""
    v = val.strip().upper()
    if v in ("BUY", "SELL", "HOLD"):
        return v
    return ""


def _extract_ticker_votes(signals: dict, ticker: str) -> dict[str, str]:
    """Collect {signal_name: action} for a given triggered ticker.

    Pulls from both ``enhanced_signals[name].action`` and the various
    ``*_action`` keys parked in ``extra`` (e.g. ``ministral_action``,
    ``qwen3_action``, ``funding_action``, ``volume_action``).
    """
    out: dict[str, str] = {}
    sig = signals.get(ticker) if isinstance(signals, dict) else None
    if not isinstance(sig, dict):
        return out
    enh = sig.get("enhanced_signals") or {}
    if isinstance(enh, dict):
        for name, payload in enh.items():
            if isinstance(payload, dict):
                a = _norm_action(payload.get("action"))
                if a:
                    out[name] = a
    extra = sig.get("extra") or {}
    if isinstance(extra, dict):
        for key, val in extra.items():
            if not isinstance(key, str) or not key.endswith("_action"):
                continue
            name = key[: -len("_action")]
            a = _norm_action(val)
            if a:
                # don't overwrite enhanced_signals entries with the same name
                out.setdefault(name, a)
    return out


def _load_accuracy_ranking(
    accuracy_cache: dict | None,
    accuracy_cache_path: str | None,
) -> list[str]:
    """Return signal names sorted by 1d accuracy (descending)."""
    cache = accuracy_cache
    if cache is None:
        path = accuracy_cache_path or DEFAULT_ACCURACY_CACHE_PATH
        cache = load_json(path, default={}) or {}
    one_d = cache.get("1d") if isinstance(cache, dict) else None
    if not isinstance(one_d, dict):
        return []
    ranked: list[tuple[str, float, int]] = []
    for name, payload in one_d.items():
        if not isinstance(payload, dict):
            continue
        acc = payload.get("accuracy")
        samples = payload.get("samples") or payload.get("total") or 0
        if not isinstance(acc, (int, float)):
            continue
        try:
            ranked.append((str(name), float(acc), int(samples)))
        except (TypeError, ValueError):
            continue
    # Require some history to avoid noise; sort by accuracy desc.
    ranked = [r for r in ranked if r[2] >= 30]
    ranked.sort(key=lambda t: t[1], reverse=True)
    return [r[0] for r in ranked]


def _top5_split(
    signals: dict,
    ticker: str,
    accuracy_cache: dict | None,
    accuracy_cache_path: str | None,
) -> tuple[bool, int, int]:
    """Return (split, n_buy, n_sell) over the top-5 most reliable signals
    that have a non-HOLD vote on this ticker."""
    if not ticker:
        return False, 0, 0
    votes = _extract_ticker_votes(signals, ticker)
    if not votes:
        return False, 0, 0
    ranking = _load_accuracy_ranking(accuracy_cache, accuracy_cache_path)
    actionable: list[str] = []
    for name in ranking:
        v = votes.get(name)
        if v in ("BUY", "SELL"):
            actionable.append(v)
        if len(actionable) >= 5:
            break
    # Fallback: if accuracy ranking missing, use whatever votes we have
    if not ranking:
        actionable = [v for v in votes.values() if v in ("BUY", "SELL")][:5]
    n_buy = sum(1 for a in actionable if a == "BUY")
    n_sell = sum(1 for a in actionable if a == "SELL")
    return (n_buy >= 2 and n_sell >= 2), n_buy, n_sell


def _ticker_held(ticker: str) -> bool:
    """Return True if either Patient or Bold has shares > 0 in ticker."""
    if not ticker:
        return False
    for fname in ("portfolio_state.json", "portfolio_state_bold.json"):
        state = load_json(str(DATA_DIR / fname), default={}) or {}
        holdings = state.get("holdings") or {}
        h = holdings.get(ticker)
        if isinstance(h, dict) and (h.get("shares") or 0) > 0:
            return True
    return False


_TICKER_RE = re.compile(r"\b([A-Z]{2,5}(?:-[A-Z]{3})?)\b")
_TICKER_BLOCKLIST = {
    "BUY", "SELL", "HOLD", "USD", "EUR", "SEK",
    "ATR", "RSI", "BB", "MA", "TF",
}


def _parse_ticker(reason: str) -> str:
    """Scan full reason string for a ticker pattern, not just the first
    token. 2026-05-15: previous first-token logic broke for reasons like
    "post-trade BTC-USD ..." (head=post-trade)."""
    if not isinstance(reason, str):
        return ""
    for m in _TICKER_RE.finditer(reason):
        tok = m.group(1)
        if tok in _TICKER_BLOCKLIST:
            continue
        # Prefer hyphenated forms (BTC-USD, XAU-USD) when present.
        if "-" in tok:
            return tok
        # Plain symbols (e.g. MSTR) are accepted only if not blocklisted.
        # Continue scanning in case a hyphenated form appears later.
        plain_candidate = tok
        # Look ahead for a hyphenated match later in the string.
        rest = reason[m.end():]
        later = _TICKER_RE.search(rest)
        if later and "-" in later.group(1) and later.group(1) not in _TICKER_BLOCKLIST:
            return later.group(1)
        return plain_candidate
    return ""


def should_escalate_to_claude(
    reasons: list[str] | Iterable[str],
    tier: int,
    signals: dict,
    accuracy_cache: dict | None = None,
    state_path: str | None = None,
    accuracy_cache_path: str | None = None,
    escalate_drawdown_pct: float = 5.0,
    current_drawdown_patient: float | None = None,
    current_drawdown_bold: float | None = None,
) -> tuple[bool, str]:
    """Decide whether a trigger goes to Claude (True) or autonomous (False).

    Returns (escalate, why). ``why`` is a short tag suitable for logs.
    """
    reasons_list = list(reasons or [])

    # 1. Tier 3 always escalates.
    if tier == 3:
        return True, "tier3"

    # 5. Post-trade triggers always escalate (cheap substring check).
    for r in reasons_list:
        if isinstance(r, str) and "post-trade" in r.lower():
            # try to surface the strategy name if present
            tag = "post_trade"
            for token in ("Patient", "Bold"):
                if token in r:
                    tag = f"post_trade_{token}"
                    break
            return True, tag

    # 2. Drawdown delta.
    state = load_json(state_path or DEFAULT_STATE_PATH, default={}) or {}
    last_p = state.get("drawdown_patient")
    last_b = state.get("drawdown_bold")
    if current_drawdown_patient is not None and isinstance(last_p, (int, float)):
        delta_p = current_drawdown_patient - float(last_p)
        if abs(delta_p) > float(escalate_drawdown_pct):
            sign = "+" if delta_p >= 0 else "-"
            return True, f"drawdown_{sign}{abs(delta_p):.1f}pct"
    if current_drawdown_bold is not None and isinstance(last_b, (int, float)):
        delta_b = current_drawdown_bold - float(last_b)
        if abs(delta_b) > float(escalate_drawdown_pct):
            sign = "+" if delta_b >= 0 else "-"
            return True, f"drawdown_{sign}{abs(delta_b):.1f}pct_bold"

    # 3 + 4. Per-ticker checks.
    for r in reasons_list:
        if not isinstance(r, str):
            continue
        ticker = _parse_ticker(r)
        if not ticker:
            continue

        # 4. Held position + SELL-side flip.
        low = r.lower()
        sell_flip = (
            "->sell" in low
            or "flipped buy->hold" in low
            or "flipped buy->sell" in low
        )
        if sell_flip and _ticker_held(ticker):
            return True, f"held_sell_flip_{ticker}"

        # 3. Top-5 reliable signals split.
        split, nb, ns = _top5_split(signals, ticker, accuracy_cache, accuracy_cache_path)
        if split:
            return True, f"top5_split_{ticker}_{nb}B_{ns}S"

    return False, "autonomous"


def record_decision_snapshot(
    drawdown_patient: float,
    drawdown_bold: float,
    state_path: str | None = None,
) -> None:
    """Persist drawdown snapshot for the next escalation comparison."""
    try:
        atomic_write_json(
            state_path or DEFAULT_STATE_PATH,
            {
                "drawdown_patient": float(drawdown_patient),
                "drawdown_bold": float(drawdown_bold),
            },
        )
    except Exception:
        logger.exception("escalation_router: failed to persist snapshot")
