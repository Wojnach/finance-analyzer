"""Shared text helpers for human-readable notifications."""

from __future__ import annotations


_THESIS_STATUS_LABELS = {
    "THREATENED": "Thesis threatened",
    "MIXED": "Thesis mixed",
    "INTACT": "Thesis intact",
    "NEUTRAL": "Thesis neutral",
}


def humanize_ticker(ticker: str | None) -> str:
    text = str(ticker or "")
    return text.replace("-USD", "").replace("-", " ").strip()


def humanize_thesis_status(status: str | None) -> str:
    key = str(status or "").upper()
    return _THESIS_STATUS_LABELS.get(key, str(status or "").replace("_", " ").title())


def format_vote_summary(buy_count: int, sell_count: int, hold_count: int | None = None) -> str:
    parts = [f"{int(buy_count)} buy", f"{int(sell_count)} sell"]
    if hold_count is not None:
        parts.append(f"{int(hold_count)} hold")
    return " / ".join(parts)


def format_confidence(confidence: float | int | None) -> str:
    if confidence is None:
        return ""
    return f"{int(round(float(confidence) * 100))}% confidence"


def format_fear_greed(value: int | float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"Fear & Greed {value}"


def format_portfolio_context(
    patient_total: float,
    patient_pnl: float,
    bold_total: float,
    bold_pnl: float,
    *,
    bold_holdings: str = "",
    consensus_accuracy: float | None = None,
) -> str:
    parts = [
        f"Patient portfolio {patient_total / 1000:.0f}K SEK ({patient_pnl:+.0f}%)",
        f"Bold portfolio {bold_total / 1000:.0f}K SEK ({bold_pnl:+.0f}%){bold_holdings}",
    ]
    if consensus_accuracy is not None:
        parts.append(f"Consensus accuracy {int(round(consensus_accuracy * 100))}%")
    return "_" + " · ".join(parts) + "_"


def format_tier_footer(label: str, tier: int, check_count: int, time_label: str) -> str:
    return f"_{label} Tier {tier} · Check #{check_count} · {time_label}_"

