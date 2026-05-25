"""Write the morning briefing JSON combining all research phases."""
import json
import datetime

briefing = {
    "date": "2026-05-26",
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "market_outlook": "neutral",
    "confidence": 0.55,
    "outlook_reasoning": "Memorial Day thin liquidity. US-Iran deal binary risk. Thursday PCE/GDP data heavy. BTC testing $80K resistance. Metals in consolidation. Ranging regime across all instruments.",
    "key_levels": {
        "XAG-USD": {"support": 73.0, "resistance": 83.0, "current": 77.5, "note": "GSR compressed to 55:1, 6th year supply deficit"},
        "XAU-USD": {"support": 4441, "resistance": 4577, "current": 4562, "note": "Central banks buying 244t Q1, structural floor"},
        "BTC-USD": {"support": 75000, "resistance": 80000, "current": 77215, "note": "Broke descending channel, testing $80K barrier"},
        "ETH-USD": {"support": 2085, "resistance": 2170, "current": 2128, "note": "RSI 36 near oversold, BTC dominance at 60%"},
        "MSTR": {"support": 146, "resistance": 170, "current": 160, "note": "Gap risk Tuesday from BTC weekend move"},
    },
    "trade_ideas": [
        {
            "instrument": "XAU-USD",
            "direction": "HOLD",
            "rationale": "Gold at $4,562, near resistance $4,577. System consensus HOLD (8 buy vs 7 sell). Iran deal binary: deal = haven unwind dip, no deal = $4,800+. Wait for clarity.",
            "confidence": 0.45,
        },
        {
            "instrument": "XAG-USD",
            "direction": "HOLD",
            "rationale": "Silver at $77.5, consensus HOLD (9 buy vs 4 sell). GSR at 55:1 already compressed. Supply deficit bullish long-term but near-term range-bound. No edge in signals (49.6% accuracy).",
            "confidence": 0.40,
        },
        {
            "instrument": "BTC-USD",
            "direction": "HOLD",
            "rationale": "BTC at $77.2K, consensus HOLD (10 buy vs 3 sell, BUY-leaning). $80K is THE level. Break above = target $82-85K. Memorial Day thin liquidity = wait. Best accuracy of our instruments (52.7%).",
            "confidence": 0.50,
        },
        {
            "instrument": "ETH-USD",
            "direction": "HOLD",
            "rationale": "ETH at $2,128, consensus HOLD (9 buy vs 7 sell, split). RSI 36 approaching oversold but BTC dominance at 60% suppressing alts. No edge (49.8% accuracy). Wait for BTC breakout to pull ETH.",
            "confidence": 0.35,
        },
    ],
    "system_improvements_implemented": [
        "Per-ticker signal re-enables: williams_vix_fix for XAU (76.5%) and XAG (60.9%), realized_skewness for XAU (60.3%), credit_spread_risk for BTC/ETH (57.4%)",
        "Per-ticker signal disables: statistical_jump_regime for XAU (50.2%), cubic_trend_persistence for XAG (41.6%), realized_skewness for XAG (42.9%), crypto_evrp for BTC (gate to ETH-only)",
        "Weight boosts: drift_regime_gate 1.4x (68.1% star), williams_vix_fix 1.3x (61.5%)",
        "Weight penalties: statistical_jump_regime 0.7x (regressing), crypto_evrp 0.6x (regressing)",
    ],
    "system_improvements_proposed": [
        {"title": "Wire walk-forward weights into live consensus", "priority": 1, "effort": "1 day", "impact": "+2-4% accuracy"},
        {"title": "IC-based signal weighting (replace accuracy gating)", "priority": 2, "effort": "3 days", "impact": "+3-5% for metals/ETH"},
        {"title": "Regime-conditional signal selection", "priority": 3, "effort": "3 days", "impact": "+2-3% regime-appropriate"},
        {"title": "TIPS real yield feature for metals signals", "priority": 6, "effort": "1 day", "impact": "+2-3% XAU/XAG"},
    ],
    "risk_warnings": [
        "CRITICAL: Consensus accuracy at/below 50% for 4/5 instruments. System has no predictive edge on metals and ETH. Per-ticker gating applied tonight may improve this.",
        "Iran deal binary risk — all instruments could gap 3-5% either direction on deal/no-deal outcome.",
        "Thursday May 28 PCE + GDP: Core PCE >3.2% = hawkish shock, <3.0% = relief rally.",
        "Memorial Day thin liquidity Monday — avoid new positions until Tuesday open.",
        "Layer 2 agent timeout: 14 triggers fired with no journal entry since May 20. Agent may not be writing decisions.",
        "Avanza session expired since May 23 — needs manual BankID re-login.",
    ],
    "research_highlights": [
        "Per-ticker signal accuracy divergence is THE root cause of poor consensus. Same signal can be 76.5% for gold but 42.9% for silver.",
        "drift_regime_gate is star performer: 68.1% recent (626 sam). Best regime detector in the system.",
        "3 disabled signals outperform active ones: williams_vix_fix (61.5%), realized_skewness (64.3%), hash_ribbons (60.7%). Two now re-enabled per-ticker.",
        "Walk-forward optimizer exists (signal_weight_optimizer.py) but weights NOT wired into live consensus. Highest-ROI single improvement.",
        "Real yields (10Y TIPS) are the #1 gold predictor per multiple studies. We have FRED integration but gold_real_yield_paradox signal is disabled.",
        "MWU learning rate (eta=0.1) is too aggressive for 15 signals/30K samples. Optimal ~0.015.",
    ],
}

with open("data/morning_briefing.json", "w", encoding="utf-8") as f:
    json.dump(briefing, f, indent=2)
print("Morning briefing written")
