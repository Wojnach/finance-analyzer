"""Prophecy step 1 — context preparation (ZERO Claude tokens).

For each ENABLED instrument, gather as much of the system's stored signals,
equations, accuracy history, macro beliefs and recent research as exists, plus a
LIVE price, into a compact per-instrument context bundle the Claude run consumes.
Also seeds a deterministic ``coverage`` block so the data/equation-gap flag is
meaningful even if the Claude step is skipped or fails.

Run: ``python -m prophecy.prep [--date YYYY-MM-DD] [--dry-run]``

Reuses (no reinvention):
- ``portfolio.outcome_tracker._fetch_current_price`` + its ticker maps,
- ``portfolio.prophecy.get_context_for_layer2`` (macro beliefs, read-only),
- ``data/agent_summary.json``, ``accuracy_cache.json``, ``morning_briefing.json``,
  ``daily_research_*.json``.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime

from portfolio.file_utils import atomic_write_json, load_json

from prophecy import config as pcfg
from prophecy import strategies
from prophecy.schema import build_coverage, grade_sufficiency

logger = logging.getLogger("prophecy.prep")

# Throttle between live-price hits so a 13-instrument sweep doesn't burst the
# Binance FAPI endpoint and collide with the 60s main loop (premortem #5).
_PRICE_THROTTLE_S = 0.25


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _fetch_price(ticker: str) -> tuple[float | None, str]:
    """Live price + a source label. None price => no feed (coverage downgrade)."""
    try:
        from portfolio.outcome_tracker import (
            BINANCE_FAPI_MAP,
            BINANCE_SPOT_MAP,
            YF_MAP,
            _fetch_current_price,
        )
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("price helper import failed: %r", exc)
        return None, "import_error"

    if ticker in BINANCE_FAPI_MAP:
        source = "binance_fapi"
    elif ticker in BINANCE_SPOT_MAP:
        source = "binance_spot"
    elif ticker in YF_MAP:
        source = "price_source"  # Alpaca for stocks, yfinance fallback
    else:
        # Oil (CL=F/BZ=F) and anything else outcome_tracker doesn't map but
        # price_source can resolve (CL=F -> Binance FAPI per CLAUDE.md). Warrants
        # + Tier-2 Avanza tickers won't resolve -> None -> coverage needs_work.
        return _price_via_price_source(ticker)

    try:
        price = _fetch_current_price(ticker)
    except Exception as exc:
        logger.warning("price fetch failed for %s: %r", ticker, exc)
        return None, f"{source}_error"
    return (float(price) if price is not None else None), source


def _price_via_price_source(ticker: str) -> tuple[float | None, str]:
    """Last-resort live price via the canonical price_source (e.g. oil)."""
    try:
        from portfolio.price_source import fetch_klines
        df = fetch_klines(ticker, interval="1m", limit=1)
        if df is None or df.empty:
            df = fetch_klines(ticker, interval="1d", limit=1, period="5d")
        if df is None or df.empty:
            return None, "no_feed"
        return float(df["close"].iloc[-1]), "price_source"
    except Exception as exc:
        logger.info("price_source has no feed for %s: %r", ticker, exc)
        return None, "no_feed"


def _ticker_signals(agent_summary: dict, ticker: str) -> dict:
    sigs = (agent_summary.get("signals") or {})
    block = sigs.get(ticker)
    return block if isinstance(block, dict) else {}


def _ticker_accuracy(accuracy_cache: dict, ticker: str) -> dict:
    out = {}
    for horizon in ("1d", "3d", "5d"):
        key = f"per_ticker_consensus_{horizon}"
        block = accuracy_cache.get(key) or {}
        if isinstance(block, dict) and ticker in block:
            out[horizon] = block[ticker]
    return out


def _ticker_research(briefing: dict, deep_dive: dict, ticker: str) -> dict:
    research: dict = {}
    levels = (briefing.get("key_levels") or {}).get(ticker)
    if levels:
        research["key_levels"] = levels
    if briefing.get("summary"):
        research["market_summary"] = briefing["summary"]
    if briefing.get("market_outlook"):
        research["market_outlook"] = briefing["market_outlook"]
    for dd in (deep_dive.get("deep_dives") or []):
        if isinstance(dd, dict) and dd.get("ticker") == ticker:
            research["deep_dive"] = dd
            break
    return research


def _ticker_beliefs(belief_ctx: dict, ticker: str) -> list:
    beliefs = belief_ctx.get("beliefs") if isinstance(belief_ctx, dict) else None
    if not isinstance(beliefs, list):
        return []
    return [b for b in beliefs if isinstance(b, dict) and b.get("ticker") == ticker]


def _capability_tokens(
    ticker: str,
    price: float | None,
    stored: dict,
    agent_summary: dict,
    accuracy: dict,
    beliefs: list,
    research: dict,
) -> set[str]:
    """Which named inputs are actually available for this instrument."""
    tokens: set[str] = set()
    if price is not None:
        tokens.add("live_price")
    if stored.get("regime"):
        tokens.add("regime")
    if stored.get("rsi") is not None:
        tokens.add("rsi")
    if agent_summary.get("fear_greed") is not None:
        tokens.add("fear_greed")
    if agent_summary.get("onchain"):
        tokens.add("on_chain_btc")
    if accuracy:
        tokens.add("accuracy")
    if beliefs:
        tokens.add("macro_beliefs")
    if research:
        tokens.add("recent_research")

    enhanced = stored.get("enhanced_signals") or {}
    enh_keys = {str(k).lower() for k in (enhanced.keys() if isinstance(enhanced, dict) else [])}
    if "metals_cross_asset" in enh_keys or agent_summary.get("cross_asset_leads"):
        tokens.add("metals_cross_asset")
    if any("cot" in k for k in enh_keys):
        tokens.add("cot_positioning")
    if "crypto_macro" in enh_keys or agent_summary.get("macro"):
        tokens.add("crypto_macro")
    # warrant_parity / warrant_barrier / warrant_strike / underlying_prediction
    # are intentionally never available at prep time (no Avanza pull, underlying
    # not yet predicted) -> they stay missing -> warrants flag needs_work.
    return tokens


def _seed_coverage(instrument: str, price: float | None, tokens: set[str]) -> dict:
    pb = strategies.playbook_for(instrument)
    required = list(pb.required_inputs) if pb else []
    found = [r for r in required if r in tokens]
    missing = [r for r in required if r not in tokens]
    suff = grade_sufficiency(len(found), len(required))

    needs_avanza = any(
        m in {"warrant_parity", "warrant_barrier", "warrant_strike", "underlying_prediction"}
        for m in missing
    )
    has_eq = bool(pb and pb.equations) and (price is not None) and not needs_avanza
    note = ""
    if price is None:
        note = "no live price feed at prep time"
    elif needs_avanza:
        note = "warrant parity/barrier/strike not pulled at prep (needs Avanza)"
    return build_coverage(
        data_sufficiency=suff,
        has_proper_equation=has_eq,
        missing_inputs=missing,
        note=note,
    )


def _playbook_dict(pb: strategies.Playbook) -> dict:
    return {
        "strategy_id": pb.strategy_id,
        "asset_class": pb.asset_class,
        "underlying": pb.underlying,
        "price_model": pb.price_model,
        "signal_emphasis": pb.signal_emphasis,
        "equations": pb.equations,
        "web_questions": pb.web_questions,
        "forum_sources": pb.forum_sources,
        "special_factors": pb.special_factors,
        "required_inputs": pb.required_inputs,
    }


def build_context(date: str | None = None, *, throttle: bool = True) -> dict:
    """Assemble the per-instrument context bundle. Pure gather; no Claude."""
    date = date or _today()
    pcfg.ensure_dirs()

    # Tripwire (premortem #1): we must never disturb the unrelated macro-beliefs
    # file. We don't touch it; this just surfaces if it vanished for any reason.
    macro_beliefs_file = pcfg.DATA_DIR / "prophecy.json"
    if not macro_beliefs_file.exists():
        logger.warning("data/prophecy.json (macro beliefs) absent — unrelated to "
                       "prophecy_runs, not touched here, but flagging")

    cfg = pcfg.load_config()
    instruments = pcfg.enabled_instruments(cfg)

    agent_summary = load_json(pcfg.DATA_DIR / "agent_summary.json", default={}) or {}
    accuracy_cache = load_json(pcfg.DATA_DIR / "accuracy_cache.json", default={}) or {}
    briefing = load_json(pcfg.DATA_DIR / "morning_briefing.json", default={}) or {}
    deep_dive = load_json(pcfg.DATA_DIR / "daily_research_ticker_deep_dive.json", default={}) or {}

    try:
        from portfolio.prophecy import get_context_for_layer2
        belief_ctx = get_context_for_layer2() or {}
    except Exception as exc:
        logger.warning("macro-belief context unavailable: %r", exc)
        belief_ctx = {}

    bundle: dict[str, dict] = {}
    for inst in instruments:
        pb = strategies.playbook_for(inst)
        price, source = _fetch_price(inst)
        if throttle:
            time.sleep(_PRICE_THROTTLE_S)

        stored = _ticker_signals(agent_summary, inst)
        accuracy = _ticker_accuracy(accuracy_cache, inst)
        beliefs = _ticker_beliefs(belief_ctx, inst)
        research = _ticker_research(briefing, deep_dive, inst)
        tokens = _capability_tokens(inst, price, stored, agent_summary, accuracy, beliefs, research)
        coverage = _seed_coverage(inst, price, tokens)

        bundle[inst] = {
            "instrument": inst,
            "asset_class": pb.asset_class,
            "strategy": pb.strategy_id,
            "live_price": price,
            "price_source": source,
            "regime": stored.get("regime"),
            "playbook": _playbook_dict(pb),
            "stored_signals": stored,
            "accuracy": accuracy,
            "macro_beliefs": beliefs,
            "recent_research": research,
            "coverage_seed": coverage,
        }
        logger.info(
            "prep %-12s px=%-12s src=%-14s suff=%-12s needs_work=%s",
            inst, price, source, coverage["data_sufficiency"], coverage["needs_work"],
        )

    out = {
        "date": date,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": pcfg.model(cfg),
        "horizons": cfg.get("horizons"),
        "instruments": bundle,
    }
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Prophecy context prep (zero tokens)")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--dry-run", action="store_true", help="print summary, still writes context")
    args = ap.parse_args(argv)

    ctx = build_context(args.date)
    out_path = pcfg.context_file(ctx["date"])
    atomic_write_json(out_path, ctx)

    n = len(ctx["instruments"])
    needs = [i for i, b in ctx["instruments"].items() if b["coverage_seed"]["needs_work"]]
    print(f"\ncontext written: {out_path}")
    print(f"instruments: {n} | needs_work (data/eq gap): {len(needs)} -> {needs}")
    if args.dry_run:
        priced = sum(1 for b in ctx["instruments"].values() if b["live_price"] is not None)
        print(f"priced: {priced}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
