"""Generate morning briefing from after-hours research 2026-05-27."""
import json
import datetime
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DATA = pathlib.Path("data")

briefing = {
    "date": "2026-05-28",
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "market_outlook": "cautious_bearish",
    "confidence": 0.55,
    "outlook_reasoning": (
        "Tomorrow (May 28) is the heaviest macro day this week: GDP 2nd estimate "
        "(consensus 0.5% vs 2.0% advance = potential stagflation shock), Core PCE, "
        "jobless claims, durable goods, personal income/spending ALL at 14:30 CET. "
        "Iran peace deal unwinding geopolitical premium in metals. BTC whale accumulation "
        "diverges from ETF outflows — squeeze setup but not yet triggered. "
        "Range-bound regime across all instruments."
    ),
    "key_levels": {
        "XAG-USD": {
            "support": 73.5,
            "resistance": 76.5,
            "current": 74.84,
            "note": (
                "Bought 105.12oz at $74.39 today (RSI 21). Total 107.68oz @ $74.68 avg. "
                "GSR compressed 62:1 to 55:1 (strong silver outperformance signal). "
                "COT: 24,671 net long, commercials heavily hedged. "
                "5th year supply deficit. Solar demand 230M oz."
            ),
        },
        "XAU-USD": {
            "support": 4400,
            "resistance": 4480,
            "current": 4456,
            "note": (
                "Central banks buying +35% QoQ (243t Q1). "
                "Iran deal = geopolitical premium unwind risk. "
                "Structural floor from CB buying."
            ),
        },
        "BTC-USD": {
            "support": 73500,
            "resistance": 77000,
            "current": 75012,
            "note": (
                "Whale accumulation 270K BTC in 30 days (largest since 2013). "
                "MVRV Z-Score 0.41 (fair value zone). "
                "8-day ETF outflow ($2B+) but exchange reserves at 7-year low. "
                "63.3% short positioning — squeeze potential if $77K breaks."
            ),
        },
        "ETH-USD": {
            "support": 2000,
            "resistance": 2150,
            "current": 2052,
            "note": (
                "74% futures longs crowded. -2.94% 7d. "
                "Now-TF signals are pure noise (flips within 30 min). "
                "Wait for BTC breakout."
            ),
        },
        "MSTR": {
            "support": 148,
            "resistance": 165,
            "current": 155.70,
            "note": (
                "Trading at 0.94x mNAV (6% DISCOUNT to BTC NAV). "
                "843,738 BTC at $75,700 avg cost. "
                "Structural shift from 2-3x premiums in 2024-25. "
                "ETF options killed premium."
            ),
        },
    },
    "trade_ideas": [
        {
            "instrument": "XAG-USD",
            "direction": "HOLD",
            "rationale": (
                "Position 107.68oz @ $74.68 avg. Bought at extreme oversold (RSI 21). "
                "Mean-reversion bounce underway but medium-TF still SELL. "
                "ATR stop ~$74.25. Iran deal headwind short-term. "
                "WAIT for GDP/PCE reaction tomorrow before adding."
            ),
            "entry_zone": "N/A (holding)",
            "target": "$120 long-term",
            "stop": "$74.25 (ATR trailing)",
            "confidence": 0.55,
        },
        {
            "instrument": "BTC-USD",
            "direction": "HOLD",
            "rationale": (
                "No position. Whale accumulation + short squeeze setup BUT "
                "$77K breakout needed for confirmation. SELL signals all day "
                "were noise (5-voter, 36% density). "
                "GDP miss tomorrow = risk-off = BTC down. Wait."
            ),
            "entry_zone": "Above $77K with RVOL >1.2x",
            "target": "$85K-$100K",
            "stop": "$73.5K",
            "confidence": 0.30,
        },
        {
            "instrument": "MSTR",
            "direction": "WATCH",
            "rationale": (
                "At 6% discount to NAV — historically rare. "
                "If BTC breaks $77K, MSTR could reprice sharply upward. "
                "But trending-down regime, RSI 36, only 1 signal voter. "
                "New signal idea: mstr_mnav_discount_depth."
            ),
            "entry_zone": "$148-150 (if BTC confirms)",
            "target": "$180-200",
            "stop": "$140",
            "confidence": 0.25,
        },
    ],
    "system_improvements_implemented": [
        "Added stablecoin_supply_ratio signal (crypto-only, shadow mode)",
        "FGL adversarial review: 17 P0 findings across 8 subsystems",
        "Completed research deliverables: macro, quant, signal audit, ticker deep dives",
    ],
    "system_improvements_proposed": [
        {
            "title": "TrustTrade-style temporal consistency filter",
            "priority": 1,
            "effort": "3 days",
            "impact": "Filter Now-TF noise (ETH BUY/HOLD flips within 30 min)",
        },
        {
            "title": "Fractional Kelly + ATR vol-targeting",
            "priority": 2,
            "effort": "3 days",
            "impact": "75% growth of full Kelly, <50% max drawdown",
        },
        {
            "title": "Adaptive ATR trailing stops (regime-aware)",
            "priority": 3,
            "effort": "2 days",
            "impact": "45-65% drawdown reduction vs fixed stops",
        },
        {
            "title": "MSTR mNAV discount signal",
            "priority": 4,
            "effort": "1 day",
            "impact": "Would have caught today's 6% discount opportunity",
        },
    ],
    "risk_warnings": [
        "CRITICAL: GDP 2nd estimate + Core PCE + jobless claims ALL at 14:30 CET tomorrow. "
        "If GDP misses (0.5% vs 2.0%) + PCE hot = stagflation shock. Risk-off for all assets.",
        "Iran peace deal: 14-point framework MOU. Metals selling as geopolitical premium "
        "unwinds. If deal fails = metals bid, if signed = further metals selloff.",
        "BTC ETF outflows: 8 consecutive days, $2B+ total. But smart money (whales) "
        "accumulating aggressively — divergence could resolve violently either way.",
        "XAG COT: commercials heavily hedged at 69.2% of OI — historically signals "
        "interim top risk. Position at avg cost, no cushion for aggressive stop.",
        "Signal accuracy: 3 enabled signals (qwen3, econ_calendar, crypto_macro) badly "
        "degraded but auto-gated. momentum at 46.4% recent is borderline.",
        "Avanza session expired since May 21 — no warrant trading possible.",
    ],
    "research_highlights": [
        "Gold/silver ratio compressed from 62:1 to 55:1 in one week — strongest "
        "move in years, silver outperformance signal.",
        "MSTR at 0.94x mNAV — 6% discount to bitcoin NAV is structural shift from "
        "2-3x premiums. ETF options killed the premium.",
        "BTC 63.3% short + Put/Call premium 5.17 = extreme bearish skew. "
        "270K whale accumulation in 30 days diverges from $2B+ ETF outflows.",
        "ArXiv: Fractional Kelly + vol-targeting achieves 75% growth at <50% drawdown.",
        "ArXiv: TrustTrade selective consensus filters temporally inconsistent signals.",
        "Signal audit: momentum_cluster (rsi, momentum, mean_reversion, bb) all in same "
        "correlation group — only leader gets full weight. System handles this correctly.",
        "Shadow signals at 0% BUY / 90-100% SELL = bearish market tailwind, "
        "not genuine quality. None should be promoted from current data.",
    ],
}

DATA.joinpath("morning_briefing.json").write_text(
    json.dumps(briefing, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("Morning briefing written to data/morning_briefing.json")

try:
    from portfolio.file_utils import load_json
    config = load_json("config.json")
    from portfolio.telegram_notifications import send_telegram

    summary = "\n".join([
        "\U0001f52c MORNING BRIEFING 2026-05-28",
        "",
        "OUTLOOK: Cautious Bearish (conf 0.55)",
        "",
        "⚠️ TOMORROW MASSIVE:",
        "GDP 2nd est + Core PCE + Jobless Claims ALL 14:30 CET",
        "If GDP misses (0.5% vs 2.0%) + PCE hot = STAGFLATION SHOCK",
        "",
        "OVERNIGHT RESEARCH:",
        "• Iran 14-pt peace deal → metals geopolitical premium unwinding",
        "• BTC whale accumulation 270K/30d vs 8-day ETF outflow ($2B+)",
        "• MSTR at 0.94x mNAV (6% DISCOUNT — structural shift)",
        "• Gold/silver ratio 62→1 → 55:1 in 1 week",
        "",
        "POSITIONS:",
        "• XAG 107.68oz @ $74.68 HOLD, RSI bounce from 21",
        "  Stop $74.25, current $74.84",
        "• BTC: no position. Watch $77K breakout",
        "• Bold: DORMANT",
        "",
        "KEY LEVELS:",
        "• XAG: S $73.5 / R $76.5",
        "• BTC: S $73.5K / R $77K (squeeze if breaks)",
        "• MSTR: S $148 / R $165 (0.94x mNAV)",
        "",
        "SIGNAL HEALTH:",
        "• news_event: 69.6% recent ⬆️ (star performer)",
        "• qwen3: 36% recent ⬇️ (auto-gated)",
        "• econ_cal: 41.5% recent ⬇️ (auto-gated)",
        "• crypto_macro: 38.2% recent ⬇️ (auto-gated)",
        "",
        "TOP RISK: GDP/PCE stagflation shock tomorrow",
        "ACT: HOLD everything until macro data clears",
    ])
    send_telegram(summary, config)
    print("Telegram sent successfully")
except Exception as e:
    print(f"Telegram send failed (non-fatal): {e}")
