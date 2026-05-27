"""Write after-hours research deliverables for 2026-05-26."""
import json
import pathlib

DATA = pathlib.Path("data")

# Quant research
quant = {
    "date": "2026-05-26",
    "research_topics": [
        "Adaptive signal weighting via IC",
        "Regime-dependent signal selection (HMM)",
        "Metals-specific predictive features",
        "BTC/crypto on-chain prediction",
        "Signal correlation/redundancy pruning",
        "Walk-forward optimization",
    ],
    "findings": [
        {
            "topic": "Walk-forward weight loop disconnected",
            "finding": "train_signal_weights.py and signal_weight_optimizer.py exist but trained weights never flow into _weighted_consensus. MWU path removed as dead code.",
            "relevance_to_us": "HIGH - all infrastructure exists, just needs connection",
            "implementation_difficulty": "medium",
            "expected_impact": "high",
            "source": "Internal code audit + arxiv 2602.00080",
        },
        {
            "topic": "IC-based weighting",
            "finding": "Rolling Spearman IC captures magnitude prediction. ic_computation.py exists but only single-horizon.",
            "relevance_to_us": "HIGH - per-ticker accuracy near coin-flip, IC surfaces magnitude prediction",
            "implementation_difficulty": "medium",
            "expected_impact": "high",
            "source": "arxiv 2509.01393",
        },
        {
            "topic": "Dynamic correlation groups dead",
            "finding": "Pearson on BUY/SELL/HOLD diluted by HOLD majority. Fix: use agreement rate.",
            "relevance_to_us": "HIGH - known bug per memory/dynamic_corr_bug.md",
            "implementation_difficulty": "medium",
            "expected_impact": "medium",
            "source": "Internal bug report",
        },
        {
            "topic": "Metals features",
            "finding": "Gold/silver ratio velocity and DXY momentum strongest predictors. Already computed but buried in composite signals.",
            "relevance_to_us": "HIGH - XAG accuracy worst at 49.6%",
            "implementation_difficulty": "easy",
            "expected_impact": "medium",
            "source": "CME Group 2026 outlook",
        },
        {
            "topic": "HMM regime detection",
            "finding": "3-state HMM would formalize drift_regime_gate (68.1% recent). hmmlearn available.",
            "relevance_to_us": "MEDIUM - drift_regime_gate already captures most of this",
            "implementation_difficulty": "medium",
            "expected_impact": "medium",
            "source": "quantinsti.com",
        },
        {
            "topic": "BTC on-chain disaggregation",
            "finding": "MVRV best for 5d+, SOPR for 1d reversions, netflow for intraday. Monolithic signal averages away horizon-specific edge.",
            "relevance_to_us": "MEDIUM - BTC accuracy 52.8%, on-chain 60% but 45 samples",
            "implementation_difficulty": "medium",
            "expected_impact": "medium",
            "source": "sciencedirect.com 2025 study",
        },
    ],
    "recommended_improvements": [
        {
            "title": "Close walk-forward weight loop",
            "description": "Feed trained weights from signal_weight_optimizer.py into _weighted_consensus",
            "priority": 1,
            "effort_days": 2,
            "files_affected": ["portfolio/signal_engine.py", "portfolio/train_signal_weights.py"],
        },
        {
            "title": "Rolling per-horizon IC weighting",
            "description": "Replace accuracy-only weighting with EWMA Spearman IC per signal/ticker/horizon",
            "priority": 2,
            "effort_days": 3,
            "files_affected": ["portfolio/ic_computation.py", "portfolio/signal_engine.py"],
        },
        {
            "title": "Fix dynamic correlation groups",
            "description": "Use agreement rate instead of Pearson. Known bug.",
            "priority": 3,
            "effort_days": 2,
            "files_affected": ["portfolio/signal_engine.py"],
        },
        {
            "title": "Extract gold/silver ratio velocity as standalone signal",
            "description": "Already computed in metals_cross_assets.py. Deploy as shadow signal.",
            "priority": 4,
            "effort_days": 1,
            "files_affected": ["portfolio/signals/gs_ratio_velocity.py"],
        },
    ],
}
DATA.joinpath("daily_research_quant.json").write_text(json.dumps(quant, indent=2))

# Signal audit
audit = {
    "date": "2026-05-26",
    "top_signals": [
        {"name": "drift_regime_gate", "accuracy_1d": 59.3, "accuracy_recent": 67.2, "samples": 1553, "note": "Best signal. Improving. SELL 71.5%."},
        {"name": "qwen3", "accuracy_1d": 59.7, "accuracy_recent": 46.8, "samples": 3987, "note": "SELL 72.7% but recent degrading"},
        {"name": "fear_greed", "accuracy_1d": 58.6, "accuracy_recent": 0.0, "samples": 10232, "note": "Robust contrarian BUY. Zero recent = rare activation."},
        {"name": "bb", "accuracy_1d": 54.9, "accuracy_recent": 57.2, "samples": 9165, "note": "Reliable and improving. SELL 58.3%."},
    ],
    "worst_enabled_signals": [
        {"name": "crypto_evrp", "note": "FIXED: formally disabled (43.4% recent)"},
        {"name": "news_event", "note": "BUY accuracy 20.5%, SELL improving to 61.3% recent"},
        {"name": "momentum", "note": "52.9% all-time, 48.8% recent. Coin-flip."},
    ],
    "p0_fixes_applied": [
        "Removed stale credit_spread_risk override for BTC/ETH",
        "Formally disabled crypto_evrp (43.4% recent, gate oscillation)",
    ],
    "shadow_signal_issue": "20+ at 0 samples: 13 need network I/O, 8 recently added, 3 LLM throttled",
    "recommendations": [
        "P1: Monitor crypto_macro for formal disable (46.5% recent)",
        "P2: Measure RSI/mean_reversion/momentum correlation",
        "P2: Fix macro_external cluster leader selection",
        "P3: Watch amihud_illiquidity (75.4%, SELL 90%) for promotion",
        "P3: Watch vwap_zscore_mr (67.8%, 87 sam) for promotion",
    ],
}
DATA.joinpath("daily_research_signal_audit.json").write_text(json.dumps(audit, indent=2))

# Ticker deep dive
ticker_dive = {
    "date": "2026-05-26",
    "tickers_analyzed": ["XAG-USD", "BTC-USD"],
    "deep_dives": [
        {
            "ticker": "XAG-USD",
            "key_predictive_features": [
                "Gold/silver ratio velocity",
                "DXY 5d momentum (inverse)",
                "Copper/gold ratio",
                "COT positioning",
                "ETF flows",
            ],
            "cross_asset_correlations": {
                "DXY": "strong inverse",
                "copper": "positive",
                "gold": "strong positive",
            },
            "current_accuracy": "49.6% consensus, 67.3% metals 3h",
            "recommended_new_signals": ["gs_ratio_velocity", "dxy_momentum"],
        },
        {
            "ticker": "BTC-USD",
            "key_predictive_features": [
                "MVRV Z-Score (macro cycle)",
                "Exchange netflow (short-term)",
                "SOPR (1d reversion)",
                "ETF flows (6-day outflow streak)",
            ],
            "current_accuracy": "52.8% consensus",
            "recommended_new_signals": [
                "Disaggregate on-chain into MVRV/SOPR/netflow/NUPL sub-signals"
            ],
        },
    ],
}
DATA.joinpath("daily_research_ticker_deep_dive.json").write_text(json.dumps(ticker_dive, indent=2))

print("All research deliverables written successfully")
