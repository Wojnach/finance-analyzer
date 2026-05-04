"""Per-(ticker, signal) state-change timestamps for the dashboard heatmap.

Pure helper: no I/O. Caller is responsible for reading/writing
`data/signal_state_since.json` via atomic helpers.

Invariant: for every (ticker, signal) pair in the *current* votes payload,
the returned payload contains a `{"vote": ..., "since": <iso>}` entry.
- If the previous payload had the same vote, `since` is preserved.
- If the vote changed, was missing, or this is cold-start, `since = now_iso`.

Disabled / N/A votes are tracked the same as any other value — the dashboard
suppresses the badge for cells already styled `cell--disabled`, so we don't
need to filter here. Tracking them keeps the helper trivially monotonic.

Wired from `portfolio.reporting.write_agent_summary`, which is the single
writer of the displayed `_votes` matrix; consuming the helper anywhere else
would race against the loop's per-cycle update.
"""

from __future__ import annotations

from typing import Any


def update_state_since(
    prev: dict[str, Any] | None,
    current_votes: dict[str, dict[str, str]],
    now_iso: str,
) -> dict[str, Any]:
    """Return a new state-since payload reflecting current votes vs prev.

    Args:
        prev: previous payload, or None / empty dict on cold start. Expected
            shape: {"updated_at": str, "votes": {ticker: {signal: {"vote": str, "since": str}}}}.
            Anything else is treated as cold start (all `since = now_iso`).
        current_votes: {ticker: {signal: "BUY"|"SELL"|"HOLD"|...}}.
        now_iso: ISO-8601 timestamp string used for new / changed entries.

    Returns:
        New payload, same shape as prev. Tickers / signals not in
        current_votes are dropped (handles ticker churn).
    """
    prev_votes = (prev or {}).get("votes") if isinstance(prev, dict) else None
    if not isinstance(prev_votes, dict):
        prev_votes = {}

    new_votes: dict[str, dict[str, dict[str, str]]] = {}
    for ticker, sigs in (current_votes or {}).items():
        if not isinstance(sigs, dict):
            continue
        prev_ticker = prev_votes.get(ticker) if isinstance(prev_votes.get(ticker), dict) else {}
        out_ticker: dict[str, dict[str, str]] = {}
        for sig_name, vote in sigs.items():
            vote_str = str(vote or "HOLD").upper()
            prev_entry = prev_ticker.get(sig_name) if isinstance(prev_ticker, dict) else None
            if (
                isinstance(prev_entry, dict)
                and prev_entry.get("vote") == vote_str
                and isinstance(prev_entry.get("since"), str)
            ):
                since = prev_entry["since"]
            else:
                since = now_iso
            out_ticker[sig_name] = {"vote": vote_str, "since": since}
        new_votes[ticker] = out_ticker

    return {"updated_at": now_iso, "votes": new_votes}
