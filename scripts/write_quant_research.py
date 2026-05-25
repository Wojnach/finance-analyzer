"""Write full quant research findings from 2026-05-25 after-hours session."""
import json
import datetime

research = {
    "date": "2026-05-25",
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "research_topics": [
        "Signal combination methods (IC-weighted, stacking, meta-learning)",
        "Regime-adaptive signal selection",
        "Walk-forward signal reweighting",
        "Silver/gold prediction features",
        "Crypto prediction (on-chain + market data)",
        "Signal correlation and redundancy pruning",
        "Position sizing under uncertainty",
    ],
    "findings": [
        {
            "topic": "Signal combination methods",
            "finding": "IC-weighted ensembles: weight signals by rolling Information Coefficient (rank correlation of signal value vs forward returns). IC above 0.05 indicates meaningful predictive power. Max Information Ratio weighting consistently outperforms equal-weight. Signal can be 51% directionally accurate but have meaningful IC if magnitude predicts return magnitude.",
            "relevance_to_us": "System uses accuracy-weighted voting. IC-weighted would be strictly superior. LinearFactorModel and SignalWeightOptimizer infrastructure exists but not wired into live consensus path.",
            "implementation_difficulty": "medium",
            "expected_impact": "high",
            "source": "FactSet IC weighting research, arxiv 2509.01393",
        },
        {
            "topic": "Signal combination methods",
            "finding": "MWU learning rate: our SignalWeightManager uses eta=0.1. Optimal eta = sqrt(ln(N)/T) with N=15 signals, T=30K samples gives eta ~0.013. Current rate too aggressive, causes weight oscillation.",
            "relevance_to_us": "Direct fix: reduce eta from 0.1 to 0.015 in signal_weights.py. MWU weights computed but normalized weights never consumed by _weighted_consensus.",
            "implementation_difficulty": "easy",
            "expected_impact": "medium",
            "source": "Hedge algorithm theory (Freund & Schapire 1997)",
        },
        {
            "topic": "Regime-adaptive signal selection",
            "finding": "HMM-based regime detection with per-regime signal whitelist. Fit GaussianHMM (3-4 states) on [log_returns, realized_vol, volume_ratio]. Per-regime, only activate signals with IC > 0.02. Refit monthly on rolling 6-month window.",
            "relevance_to_us": "drift_regime_gate (68.1% recent) should gate which signals participate, not just apply a multiplier. REGIME_GATED_SIGNALS exists but is manually curated, not data-driven.",
            "implementation_difficulty": "hard",
            "expected_impact": "high",
            "source": "arxiv 2508.11338, quantinsti.com regime-adaptive",
        },
        {
            "topic": "Walk-forward signal reweighting",
            "finding": "Walk-forward optimizer already exists in signal_weight_optimizer.py with 30d train / 7d test windows. Results saved to data/models/walkforward_results.json but NEVER consumed by _weighted_consensus(). Highest-ROI single change.",
            "relevance_to_us": "Wire walkforward_results.json into consensus: if avg_oos_corr > 0.02, use recommended_weights as multipliers. Fallback to accuracy-only if OOS correlation too low.",
            "implementation_difficulty": "easy",
            "expected_impact": "high",
            "source": "Existing codebase analysis",
        },
        {
            "topic": "Silver/gold prediction features",
            "finding": "Top features for gold/silver: (1) US 10Y TIPS real yield (strongest predictor), (2) DXY (weakening since 2022), (3) COT positioning, (4) Gold ETF flows, (5) Central bank purchases. Real yields now better predictor than DXY alone.",
            "relevance_to_us": "gold_real_yield_paradox signal is DISABLED. FRED integration exists. Re-enabling with walk-forward validation could add 2-3% accuracy for XAU/XAG.",
            "implementation_difficulty": "easy",
            "expected_impact": "high",
            "source": "JP Morgan gold research, sciencedirect precious metals ML",
        },
        {
            "topic": "Crypto prediction",
            "finding": "Combined on-chain + market models achieve 84.3% for macro cycle positioning. Key features: MVRV Z-Score (current 1.32), SOPR, exchange netflow (6yr lows), LTH supply ratio (78%). Funding rates mean-reverting and predictive at extremes.",
            "relevance_to_us": "onchain signal (60% 1d) already uses MVRV/SOPR. Should weight MORE heavily on longer horizons (3d/5d). Funding rate should be FILTER not standalone voter.",
            "implementation_difficulty": "easy",
            "expected_impact": "medium",
            "source": "Bitcoin ML research, sciencedirect crypto forecasting",
        },
        {
            "topic": "Signal correlation and redundancy",
            "finding": "Agreement rate (fraction of time two signals vote same direction) > 80% indicates functional redundancy. After pruning redundant signals, re-optimize weights. Likely redundant pairs: (RSI, mean_reversion), (BB, mean_reversion), (momentum, RSI).",
            "relevance_to_us": "_compute_agreement_rate() already exists in signal_engine.py. Pruning from 15 to 10-12 independent signals could improve consensus by eliminating double-counted votes.",
            "implementation_difficulty": "easy",
            "expected_impact": "medium",
            "source": "arxiv 2509.01393, 2509.25055",
        },
        {
            "topic": "Position sizing under uncertainty",
            "finding": "Full Kelly optimal only with perfect estimates. With 4/5 instruments at <=50% accuracy, full Kelly = zero position. Quarter-Kelly retains 75% growth rate, halves variance. MacLean (2004): most successful institutions use quarter-to-half Kelly.",
            "relevance_to_us": "kelly_sizing.py exists. Enforce quarter-Kelly max. Hard rule: if consensus accuracy <52% on trailing 30d, cap position at 2% portfolio.",
            "implementation_difficulty": "easy",
            "expected_impact": "medium",
            "source": "Kelly criterion literature, MacLean et al. 2004",
        },
    ],
    "recommended_improvements": [
        {
            "title": "Wire walk-forward weights into live consensus",
            "description": "Load recommended_weights from data/models/walkforward_results.json into _weighted_consensus(). Fallback if OOS corr < 0.02.",
            "priority": 1,
            "effort_days": 1,
            "expected_accuracy_improvement": "+2-4% consensus accuracy",
            "files_affected": ["portfolio/signal_engine.py", "portfolio/signal_weight_optimizer.py"],
        },
        {
            "title": "IC-based signal weighting",
            "description": "Replace accuracy gating with rolling 30d rank-IC per signal per ticker. Weight by IC * sqrt(ICIR).",
            "priority": 2,
            "effort_days": 3,
            "expected_accuracy_improvement": "+3-5% for metals/ETH",
            "files_affected": ["portfolio/signal_engine.py", "portfolio/accuracy_stats.py"],
        },
        {
            "title": "Regime-conditional signal selection",
            "description": "Use drift_regime_gate to partition history into regimes. Per-regime, only activate signals with IC > 0.02.",
            "priority": 3,
            "effort_days": 3,
            "expected_accuracy_improvement": "+2-3%",
            "files_affected": ["portfolio/signal_engine.py", "portfolio/accuracy_stats.py"],
        },
        {
            "title": "TIPS real yield feature for metals",
            "description": "Add DFII10 (10Y TIPS yield) from FRED to metals_cross_assets.py. Audit gold_real_yield_paradox for re-enable.",
            "priority": 4,
            "effort_days": 1,
            "expected_accuracy_improvement": "+2-3% XAU/XAG",
            "files_affected": ["portfolio/metals_cross_assets.py", "portfolio/signals/gold_real_yield_paradox.py"],
        },
        {
            "title": "Reduce MWU eta and wire weights into consensus",
            "description": "Reduce eta from 0.1 to 0.015. Wire MWU normalized weights into _weighted_consensus().",
            "priority": 5,
            "effort_days": 1,
            "expected_accuracy_improvement": "+0.5-1%",
            "files_affected": ["portfolio/signal_weights.py", "portfolio/signal_engine.py"],
        },
    ],
}

with open("data/daily_research_quant.json", "w", encoding="utf-8") as f:
    json.dump(research, f, indent=2)
print("Quant research written")
