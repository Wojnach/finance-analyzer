"""Deep focus probability-and-range analysis for selected instruments.

Provides a one-shot report (console + Telegram via message_store) with:
- Directional probability at 3h/1d/3d horizons
- Near-close estimated price range (ATR + forecast blend)
- Explicit handling for missing instruments (e.g., MSTR not in current Layer 1 set)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from portfolio.file_utils import load_json, load_jsonl
from portfolio.market_timing import MARKET_OPEN_HOUR, _market_close_hour_utc
from portfolio.message_store import send_or_store

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_runtime_root() -> Path:
    """Resolve runtime root (works in main repo and git worktrees).

    Priority:
    1) Current repo root if it has config.json + data/
    2) First parent root that has config.json + data/
    3) BASE_DIR fallback
    """
    candidates = [BASE_DIR, *BASE_DIR.parents]
    for root in candidates:
        if (root / "config.json").exists() and (root / "data").exists():
            return root
    return BASE_DIR


RUNTIME_ROOT = _resolve_runtime_root()
DATA_DIR = RUNTIME_ROOT / "data"
SUMMARY_FILE = DATA_DIR / "agent_summary.json"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
CONFIG_FILE = RUNTIME_ROOT / "config.json"

DEFAULT_FOCUS_TICKERS = ["XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD", "MSTR"]

logger = logging.getLogger("portfolio.focus_analysis")


def _format_price(value: float) -> str:
    if value >= 10000:
        return "$" + f"{value:,.0f}"
    if value >= 1000:
        return "$" + f"{value:,.2f}"
    return "$" + f"{value:.2f}"


def normalize_ticker(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    aliases = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "XAU": "XAU-USD",
        "XAG": "XAG-USD",
        "GOLD": "XAU-USD",
        "SILVER": "XAG-USD",
        "MICROSTRATEGY": "MSTR",
    }
    return aliases.get(t, t)


def hours_to_us_close(now: datetime | None = None) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return 0.0
    close_hour = _market_close_hour_utc(now)
    if now.hour >= close_hour:
        return 0.0
    if now.hour < MARKET_OPEN_HOUR:
        return float(close_hour - MARKET_OPEN_HOUR)
    return max(0.0, close_hour - now.hour - now.minute / 60.0)


def latest_journal_price(ticker: str, default: float = 0.0) -> float:
    entries = load_jsonl(JOURNAL_FILE, limit=400)
    for entry in reversed(entries):
        px = entry.get("prices", {}).get(ticker)
        if isinstance(px, (int, float)) and px > 0:
            return float(px)
    return default


def extract_focus_probabilities(summary: dict, tickers: list[str]) -> dict:
    probs = {}
    existing = summary.get("focus_probabilities", {}) or {}
    for t in tickers:
        if t in existing:
            probs[t] = existing[t]

    missing = [t for t in tickers if t not in probs and t in summary.get("signals", {})]
    if missing:
        try:
            from portfolio.ticker_accuracy import get_focus_probabilities

            signal_map = {k: v for k, v in summary.get("signals", {}).items() if k in missing}
            computed = get_focus_probabilities(
                missing, signal_map, horizons=["3h", "1d", "3d"], days=7
            )
            probs.update(computed or {})
        except Exception:
            pass
    return probs


def estimate_near_close_range(
    price: float,
    atr_pct: float,
    prob_3h: dict | None,
    forecast_1h_pct: float,
    hours_left: float,
) -> tuple[float | None, float | None, float]:
    if not isinstance(price, (int, float)) or price <= 0:
        return None, None, 0.0

    h = max(1.0, min(hours_left if hours_left > 0 else 3.0, 6.0))
    atr_pct = float(atr_pct or 0.0)

    # Volatility envelope from ATR scaled to near-close horizon.
    base_move_pct = atr_pct * math.sqrt(h / 6.0) if atr_pct > 0 else 0.6 * math.sqrt(h / 3.0)

    # Forecast contribution from Chronos 1h projected over near horizon.
    f1h = abs(float(forecast_1h_pct or 0.0))
    forecast_move_pct = f1h * min(h, 3.0)

    move_pct = max(base_move_pct, forecast_move_pct, 0.35)

    p_up = float((prob_3h or {}).get("probability", 0.5))
    drift_pct = (p_up - 0.5) * 2.0 * move_pct * 0.35

    low = price * (1.0 + (drift_pct - move_pct) / 100.0)
    high = price * (1.0 + (drift_pct + move_pct) / 100.0)
    return low, high, move_pct


def _ticker_name(ticker: str) -> str:
    return ticker.replace("-USD", "")


def run_focus_analysis(tickers: list[str] | None = None) -> str:
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError("No agent_summary.json found. Run --report first.")

    summary = load_json(SUMMARY_FILE)
    if summary is None:
        raise FileNotFoundError("Failed to parse agent_summary.json.")
    config = load_json(CONFIG_FILE, default={})

    focus = [normalize_ticker(t) for t in (tickers or DEFAULT_FOCUS_TICKERS)]
    hours_left = hours_to_us_close()
    logger.info("focus analysis start: tickers=%s hours_to_close=%.2f", focus, hours_left)
    probs = extract_focus_probabilities(summary, focus)
    forecast = summary.get("forecast_signals", {}) or {}

    lines: list[str] = []
    lines.append(f"*DEEP FOCUS ANALYSIS* (to US close: ~{hours_left:.1f}h)")
    lines.append("")

    fallback_count = 0
    for ticker in focus:
        sig = summary.get("signals", {}).get(ticker)
        t_probs = probs.get(ticker, {})
        p3h = t_probs.get("3h", {})
        p1d = t_probs.get("1d", {})
        p3d = t_probs.get("3d", {})
        forecast_1h = (forecast.get(ticker) or {}).get("chronos_1h_pct", 0)

        if sig:
            price = float(sig.get("price_usd", 0) or 0)
            atr_pct = float(sig.get("atr_pct", 0) or 0)
            low, high, move_pct = estimate_near_close_range(
                price=price,
                atr_pct=atr_pct,
                prob_3h=p3h,
                forecast_1h_pct=forecast_1h,
                hours_left=hours_left,
            )

            d3h = "UP" if p3h.get("direction", "up") == "up" else "DN"
            d1d = "UP" if p1d.get("direction", "up") == "up" else "DN"
            d3d = "UP" if p3d.get("direction", "up") == "up" else "DN"
            p3h_pct = int(round(float(p3h.get("probability", 0.5)) * 100))
            p1d_pct = int(round(float(p1d.get("probability", 0.5)) * 100))
            p3d_pct = int(round(float(p3d.get("probability", 0.5)) * 100))
            acc_pct = int(round(float(p1d.get("accuracy", 0)) * 100)) if p1d.get("accuracy") is not None else 0
            samples = int(p1d.get("samples", 0) or 0)

            lines.append(
                f"`{_ticker_name(ticker):<5} {_format_price(price):>8}  {d3h}{p3h_pct}% 3h  {d1d}{p1d_pct}% 1d  {d3d}{p3d_pct}% 3d`"
            )
            if low is not None and high is not None:
                lines.append(
                    f"`  close range: {_format_price(low)} - {_format_price(high)} (+/-{move_pct:.2f}%)`"
                )
            lines.append(
                f"`  acc: {acc_pct}% 1d ({samples} sam) | chronos1h: {forecast_1h:+.2f}% | atr: {atr_pct:.2f}%`"
            )
        else:
            fallback_price = latest_journal_price(ticker, default=0.0)
            fallback_count += 1
            if fallback_price > 0:
                low, high, move_pct = estimate_near_close_range(
                    price=fallback_price,
                    atr_pct=1.6,
                    prob_3h={"probability": 0.5},
                    forecast_1h_pct=0.0,
                    hours_left=hours_left,
                )
                lines.append(
                    f"`{_ticker_name(ticker):<5} {_format_price(fallback_price):>8}  UP50% 3h  UP50% 1d  UP50% 3d`"
                )
                lines.append(
                    f"`  close range: {_format_price(low)} - {_format_price(high)} (fallback +/-{move_pct:.2f}%)`"
                )
                lines.append("`  note: no live Layer1 signal row; using journal-price fallback`")
            else:
                lines.append(f"`{_ticker_name(ticker):<5} no live data in agent_summary`")
        lines.append("")

    lines.append(
        "_Method: probability = per-ticker signal-accuracy weighting; range = ATR + forecast blend, skewed by 3h direction._"
    )

    msg = "\n".join(lines)[:4096]
    send_or_store(msg, config, category="analysis")
    return msg
