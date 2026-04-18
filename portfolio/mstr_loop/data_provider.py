"""Build an MstrBundle each cycle — the single input object strategies receive.

A bundle is the MSTR signal snapshot (from agent_summary_compact.json)
plus the MSTR-weighted consensus score plus live price fields strategies
need for entry/exit math. Everything a strategy could want is here; no
strategy reads raw JSON.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
from typing import Any

from portfolio.file_utils import load_json

from portfolio.mstr_loop import config

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MstrBundle:
    """Everything a strategy needs to make a decision this cycle."""
    ts: str                        # ISO-8601 UTC, built at cycle start
    source_stale_seconds: float    # age of agent_summary_compact.json
    # Price
    price_usd: float               # current MSTR price from signal block
    # Consensus from signal engine
    raw_action: str                # "BUY" | "SELL" | "HOLD"
    raw_weighted_confidence: float
    rsi: float
    macd_hist: float
    bb_position: str
    regime: str
    atr_pct: float
    # Vote counts (from _buy_count / _sell_count in agent_summary)
    buy_count: int
    sell_count: int
    total_voters: int
    # Individual signal votes (filtered to what's present). Key = signal
    # module name (e.g. "ministral"), value in {"BUY","SELL","HOLD"}.
    votes: dict[str, str]
    # Monte Carlo outputs
    p_up_1d: float
    exp_return_1d_pct: float
    exp_return_3d_pct: float
    # Heatmap (Now, 12h, 2d, 7d, 1mo, 3mo, 6mo)
    heatmap: list[dict[str, str]]
    # Is the signal block marked stale by the main loop? (MSTR closed hours)
    stale: bool
    # Derived by this module using MSTR_SIGNAL_WEIGHTS
    weighted_score_long: float     # 0-1 scale, higher = stronger LONG case
    weighted_score_short: float    # 0-1 scale, higher = stronger SHORT case
    # BTC regime (2026-04-18) — MSTR is 1.5-2.5x BTC beta; if BTC is in a
    # confirmed down-trend, MSTR LONG is structurally wrong regardless of
    # MSTR's own technicals. Values: "trending-up" | "trending-down" |
    # "ranging" | "high-vol" | "unknown" (when BTC block is absent).
    btc_regime: str = "unknown"
    btc_price: float = 0.0         # spot for cross-asset awareness
    btc_rsi: float = 50.0

    def is_usable(self) -> bool:
        """True if the bundle looks fresh/valid enough to trade on."""
        if self.stale:
            return False
        if self.price_usd <= 0:
            return False
        # Staler than 5 minutes means the loop or signal engine is lagging —
        # better to skip a cycle than act on potentially wrong data.
        if self.source_stale_seconds > 300:
            return False
        return True


def _parse_vote_detail(vote_detail: str) -> dict[str, str]:
    """Parse the compacted `_vote_detail` string into a {signal: action} map.

    Format: ``"B:sig1,sig2,sig3 | S:sig4,sig5 | H:sig6"`` — each segment
    names the direction letter, colon, comma-separated signals. Signals
    not listed get an implicit HOLD. Whitespace-tolerant.
    """
    if not vote_detail:
        return {}
    out: dict[str, str] = {}
    letter_map = {"B": "BUY", "S": "SELL", "H": "HOLD"}
    for segment in vote_detail.split("|"):
        segment = segment.strip()
        if not segment or ":" not in segment:
            continue
        letter, _, names = segment.partition(":")
        action = letter_map.get(letter.strip())
        if action is None:
            continue
        for name in names.split(","):
            name = name.strip()
            if name:
                out[name] = action
    return out


def _compute_weighted_scores(votes: dict[str, str]) -> tuple[float, float]:
    """Compute weighted LONG and SHORT scores using MSTR_SIGNAL_WEIGHTS.

    Scaled to [0,1]: fraction of *active* (non-HOLD) voter weight that
    aligns with direction. HOLDs are excluded from BOTH numerator and
    denominator so the score measures conviction among voters who took a
    side, not overall activation. Activation / minimum-voters is a
    separate gate in the strategy layer (MIN_BUY_VOTERS etc.).

    Rationale (2026-04-19 backtest audit): the previous implementation
    counted HOLDs in the denominator, producing sub-0.35 scores across
    30 days of historical MSTR signals even on days when live had 0.68+.
    That live/historical mismatch was because live `_vote_detail` compacts
    out HOLDs (no dilution), while signal_log keeps the full dict (massive
    HOLD dilution). Excluding HOLDs everywhere makes the two paths
    identical and threshold semantics consistent. Strategy tests that
    feed only {"x": "BUY", "y": "BUY"} shapes still pass trivially.

    Returns (long_score, short_score). Both in [0,1], sum ≤ 1.
    """
    weights = config.MSTR_SIGNAL_WEIGHTS
    default_w = config.DEFAULT_SIGNAL_WEIGHT

    long_num = 0.0
    short_num = 0.0
    denom = 0.0
    for signal_name, vote in votes.items():
        w = weights.get(signal_name, default_w)
        if w <= 0:
            continue  # force-ignored signal
        if vote not in ("BUY", "SELL"):
            continue  # HOLD or unknown — excluded from both sides
        denom += w
        if vote == "BUY":
            long_num += w
        elif vote == "SELL":
            short_num += w
    if denom <= 0:
        return 0.0, 0.0
    return long_num / denom, short_num / denom


def build_bundle(
    agent_summary_path: str = "data/agent_summary_compact.json",
    ticker: str = None,  # type: ignore[assignment]
) -> MstrBundle | None:
    """Read agent_summary and build the MstrBundle. None on unreadable/missing."""
    import os
    import time

    t = ticker or config.MSTR_UNDERLYING

    if not os.path.exists(agent_summary_path):
        logger.debug("data_provider: agent_summary missing at %s", agent_summary_path)
        return None

    try:
        summary = load_json(agent_summary_path)
    except Exception:
        logger.warning("data_provider: agent_summary load failed", exc_info=True)
        return None

    if not isinstance(summary, dict):
        return None

    sig = summary.get("signals", {}).get(t, {})
    if not sig:
        logger.debug("data_provider: no signal block for ticker %s", t)
        return None

    mc = summary.get("monte_carlo", {}).get(t, {}) or {}
    timeframes = summary.get("timeframes", {}).get(t, []) or []
    extra = sig.get("extra", {}) or {}
    votes = extra.get("_votes") or {}
    # Compact agent_summary drops the full _votes dict to save tokens and
    # only keeps the _vote_detail string ("B:sig1,sig2 | S:sig3"). Parse it
    # as a fallback so we still get per-signal weighting.
    if not votes:
        votes = _parse_vote_detail(extra.get("_vote_detail", ""))

    long_score, short_score = _compute_weighted_scores(votes)

    try:
        source_stale_seconds = time.time() - os.path.getmtime(agent_summary_path)
    except OSError:
        source_stale_seconds = 9999.0

    # BTC cross-asset block — read the precomputed regime from the main
    # loop's signal engine. Fall back to "unknown" if BTC block missing.
    btc_sig = summary.get("signals", {}).get("BTC-USD", {}) or {}
    btc_regime = str(btc_sig.get("regime") or "unknown")
    btc_price = float(btc_sig.get("price_usd") or 0)
    btc_rsi = float(btc_sig.get("rsi") or 50)

    return MstrBundle(
        ts=datetime.datetime.now(datetime.UTC).isoformat(),
        source_stale_seconds=source_stale_seconds,
        price_usd=float(sig.get("price_usd") or 0),
        raw_action=str(sig.get("action", "HOLD")),
        raw_weighted_confidence=float(sig.get("weighted_confidence") or 0),
        rsi=float(sig.get("rsi") or 50),
        macd_hist=float(sig.get("macd_hist") or 0),
        bb_position=str(sig.get("bb_position") or ""),
        regime=str(sig.get("regime") or "unknown"),
        atr_pct=float(sig.get("atr_pct") or 0),
        buy_count=int(extra.get("_buy_count") or 0),
        sell_count=int(extra.get("_sell_count") or 0),
        total_voters=int(extra.get("_voters") or 0),
        votes=dict(votes),
        p_up_1d=float(mc.get("p_up") or 0.5),
        exp_return_1d_pct=float((mc.get("expected_return_1d") or {}).get("mean_pct") or 0),
        exp_return_3d_pct=float((mc.get("expected_return_3d") or {}).get("mean_pct") or 0),
        heatmap=list(timeframes),
        stale=bool(sig.get("stale", False)),
        weighted_score_long=long_score,
        weighted_score_short=short_score,
        btc_regime=btc_regime,
        btc_price=btc_price,
        btc_rsi=btc_rsi,
    )
