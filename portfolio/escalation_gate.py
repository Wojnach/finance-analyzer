"""Ministral pre-gate — Batch E / item 7 of docs/PLAN.md.

Final cheap classifier between ``escalation_router.should_escalate_to_claude``
(hard rule) and ``invoke_agent`` (Claude CLI subprocess). After the router
approves an escalation, this gate runs a short structured Ministral-8B
classification and may downgrade the path back to autonomous when the LLM
is confidently "not escalate".

Contract:
  - Fail-open. Any error (timeout, missing runner, parse failure, empty
    reasons) returns ``(True, 0.0, "<tag>")`` so we never silently swallow
    a trigger.
  - Logs every call to ``data/escalation_gate.jsonl`` via atomic append.
  - Runner is injectable for testing; the default lazily imports
    ``portfolio.llama_server.query_llama_server`` and asks for JSON.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import pathlib
import re
from typing import Callable

from portfolio.file_utils import atomic_append_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DEFAULT_LOG_PATH = str(DATA_DIR / "escalation_gate.jsonl")


_PROMPT_HEADER = (
    "You are a cheap classifier deciding whether a trading trigger needs "
    "an expensive LLM (Claude) or can be handled autonomously. Reply with "
    "ONLY a JSON object: "
    '{"escalate": bool, "confidence": 0.0..1.0, "why": "<short reason>"}.\n'
    "Escalate=true only if the situation looks ambiguous, conflicting, "
    "or unusual. Escalate=false for clean trend continuations, noise, "
    "or routine signal flips on instruments we do not hold.\n"
)


def _norm_action(val) -> str:
    if not isinstance(val, str):
        return ""
    v = val.strip().upper()
    return v if v in ("BUY", "SELL", "HOLD") else ""


def _top_ticker_posture(signals: dict, ticker: str, n: int = 5) -> list[str]:
    sig = signals.get(ticker) if isinstance(signals, dict) else None
    if not isinstance(sig, dict):
        return []
    rows: list[str] = []
    enh = sig.get("enhanced_signals") or {}
    if isinstance(enh, dict):
        for name, payload in enh.items():
            if not isinstance(payload, dict):
                continue
            a = _norm_action(payload.get("action"))
            if a:
                rows.append(f"{name}={a}")
            if len(rows) >= n:
                break
    return rows[:n]


def _triggered_tickers(reasons: list[str]) -> list[str]:
    out: list[str] = []
    for r in reasons or []:
        if not isinstance(r, str):
            continue
        head = r.split()[0] if r.split() else ""
        if head and "-" in head and head.replace("-", "").isalnum():
            if head not in out:
                out.append(head)
    return out


def _build_prompt(
    reasons: list[str],
    tier: int,
    signals: dict,
    held_positions: dict,
    tickers: list[str],
) -> str:
    lines: list[str] = [_PROMPT_HEADER, f"Tier: {tier}", "Reasons:"]
    for r in reasons[:8]:
        lines.append(f"- {r}")
    lines.append("Top signal posture:")
    for t in tickers[:3]:
        posture = _top_ticker_posture(signals, t, 5)
        if posture:
            lines.append(f"  {t}: " + ", ".join(posture))
    held_p = (held_positions or {}).get("patient") or []
    held_b = (held_positions or {}).get("bold") or []
    lines.append(f"Held Patient: {', '.join(held_p) if held_p else '(none)'}")
    lines.append(f"Held Bold: {', '.join(held_b) if held_b else '(none)'}")
    lines.append('\nReturn ONLY the JSON object.')
    return "\n".join(lines)


def _default_runner(prompt: str) -> str:
    """Default ministral runner. Returns raw text. Caller parses JSON."""
    from portfolio.llama_server import query_llama_server
    return query_llama_server(
        "ministral3",
        prompt,
        n_predict=128,
        temperature=0.0,
    )


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_response(text: str) -> tuple[bool, float, str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty")
    # Try direct JSON parse first; else grep first {...} block.
    candidate = text.strip()
    try:
        obj = json.loads(candidate)
    except Exception:
        m = _JSON_RE.search(candidate)
        if not m:
            raise ValueError("no_json")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("not_dict")
    if "escalate" not in obj:
        raise ValueError("missing_escalate")
    esc = bool(obj.get("escalate"))
    conf_raw = obj.get("confidence", 0.0)
    try:
        conf = float(conf_raw)
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    why = str(obj.get("why") or "").strip() or ("escalate" if esc else "not_escalate")
    return esc, conf, why


def _log_decision(
    log_path: str,
    reasons: list[str],
    tier: int,
    escalate: bool,
    confidence: float,
    why: str,
    tickers: list[str],
) -> None:
    try:
        atomic_append_jsonl(
            log_path,
            {
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "reasons": list(reasons or []),
                "tier": int(tier),
                "escalate": bool(escalate),
                "confidence": float(confidence),
                "why": str(why),
                "tickers": list(tickers or []),
            },
        )
    except Exception:
        logger.exception("escalation_gate: failed to append log row")


def should_escalate(
    reasons: list[str],
    tier: int,
    signals: dict,
    prices: dict,
    held_positions: dict,
    *,
    runner: Callable[[str], str] | None = None,
    log_path: str | None = None,
) -> tuple[bool, float, str]:
    """Run Ministral as a second-opinion classifier.

    Returns ``(escalate, confidence, why)``. Fail-open on any error:
    any unexpected exception or parse failure returns
    ``(True, 0.0, "ministral_unavailable")``.
    """
    lp = log_path or DEFAULT_LOG_PATH
    reasons_list = list(reasons or [])
    tickers = _triggered_tickers(reasons_list)

    if not reasons_list:
        _log_decision(lp, reasons_list, tier, True, 0.0, "empty_reasons", tickers)
        return True, 0.0, "ministral_unavailable"

    prompt = _build_prompt(reasons_list, tier, signals or {}, held_positions or {}, tickers)
    call = runner or _default_runner
    # 2026-05-15: explicit 10s wall-clock budget. Ministral is supposed
    # to be fast (~1-2s); if it hangs we fail open rather than block
    # the entire trigger pipeline.
    import concurrent.futures as _cf
    _ex = _cf.ThreadPoolExecutor(max_workers=1)
    try:
        _fut = _ex.submit(call, prompt)
        try:
            raw = _fut.result(timeout=10)
        except _cf.TimeoutError:
            logger.warning("escalation_gate: runner timed out (>10s) — failing open")
            _log_decision(lp, reasons_list, tier, True, 0.0, "runner_timeout", tickers)
            # Don't wait for the hung thread on shutdown.
            _ex.shutdown(wait=False, cancel_futures=True)
            return True, 0.0, "ministral_unavailable"
        finally:
            _ex.shutdown(wait=False, cancel_futures=True)
    except Exception as e:
        logger.warning("escalation_gate: runner raised %s — failing open", e)
        _log_decision(lp, reasons_list, tier, True, 0.0, f"runner_error:{type(e).__name__}", tickers)
        return True, 0.0, "ministral_unavailable"

    try:
        esc, conf, why = _parse_response(raw)
    except Exception as e:
        logger.warning("escalation_gate: parse failed (%s) — failing open", e)
        _log_decision(lp, reasons_list, tier, True, 0.0, f"parse_error:{e}", tickers)
        return True, 0.0, "ministral_unavailable"

    _log_decision(lp, reasons_list, tier, esc, conf, why, tickers)
    return esc, conf, why
