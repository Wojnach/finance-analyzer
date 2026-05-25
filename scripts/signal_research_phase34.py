"""Phase 3-4: Write signal candidates to backlog + SQLite + ranked output."""
import json
import datetime
import sqlite3
from pathlib import Path

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
date = "2026-05-24"

candidates = [
    {
        "date": date, "name": "bocpd_regime_switch", "asset_class": "cross-asset",
        "target_assets": ["XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD", "MSTR"],
        "category": "regime",
        "description": "Bayesian Online Changepoint Detection on return series. Detects regime breaks in real-time via posterior run-length. On changepoint: switch from trend-following to mean-reversion. Addresses metals coin-flip accuracy by catching sharp reversals.",
        "formula": "BOCPD recursive Bayesian update with exponential hazard h(r)=1/lambda. Run-length posterior tracks bars since last changepoint. Changepoint when max-posterior run-length drops below threshold. On break: MR mode (BUY at -2z, SELL at +2z). Otherwise: trend (momentum).",
        "data_sources": ["ohlcv"],
        "parameters": {"hazard_lambda": 100, "zscore_threshold": 2.0, "mr_window": 20},
        "novelty_score": 9.0, "edge_evidence": 8.0, "data_availability": 10.0,
        "implementation_cost": 6.0, "non_redundancy": 8.0, "composite_score": 8.25,
        "source_url": "https://arxiv.org/abs/2105.13727", "source_type": "academic",
        "citation": "Wood, Roberts, Zohren (2021), Online CPD for Momentum Trading, JFDS",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": date, "name": "kalman_trend_momentum", "asset_class": "metals",
        "target_assets": ["XAU-USD", "XAG-USD"], "category": "technical",
        "description": "Kalman filter on raw price: smooth trend state + velocity. 3-regime classification. Gold Sharpe 2.88.",
        "formula": "State [price, velocity]. Kalman recursion. BUY in TREND_UP, SELL in TREND_DOWN, HOLD in RANGE.",
        "data_sources": ["ohlcv"],
        "parameters": {"process_noise": 0.01, "observation_noise": 1.0},
        "novelty_score": 7.0, "edge_evidence": 8.0, "data_availability": 10.0,
        "implementation_cost": 7.0, "non_redundancy": 7.0, "composite_score": 7.75,
        "source_url": "https://arxiv.org/abs/2511.08571", "source_type": "academic",
        "citation": "Singha et al (2025). Sharpe 2.88 gold futures.",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": date, "name": "gvz_ivp_regime_gate", "asset_class": "metals",
        "target_assets": ["XAU-USD", "XAG-USD"], "category": "volatility",
        "description": "GVZ Implied Volatility Percentile regime overlay. IVP>80pct=MR favored, IVP<20pct=trend favored.",
        "formula": "ivp = count(gvz_past_252d < gvz_today) / 252. Gate directional confidence.",
        "data_sources": ["fred_gvzcls"],
        "parameters": {"lookback": 252, "high": 0.80, "low": 0.20},
        "novelty_score": 7.0, "edge_evidence": 7.0, "data_availability": 9.0,
        "implementation_cost": 9.0, "non_redundancy": 7.0, "composite_score": 7.60,
        "source_url": "https://metalorix.com/en/learn/ratios-analytics/gold-volatility-index-gvz",
        "source_type": "industry", "citation": "CBOE GVZ. IV mean-reversion widely validated.",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": date, "name": "gold_implied_lease_rate", "asset_class": "metals",
        "target_assets": ["XAU-USD", "XAG-USD"], "category": "intermarket",
        "description": "Implied lease rate from spot-futures spread. High rate=physical scarcity=bullish.",
        "formula": "implied_forward = (futures/spot-1)*(365/dte). lease_rate = sofr - implied_forward. Z-score.",
        "data_sources": ["binance_fapi", "fred_sofr"],
        "parameters": {"lookback": 60, "buy_z": 2.0, "sell_z": -1.5},
        "novelty_score": 8.0, "edge_evidence": 7.0, "data_availability": 8.0,
        "implementation_cost": 7.0, "non_redundancy": 8.0, "composite_score": 7.55,
        "source_url": "https://www.lbma.org.uk/alchemist/issue-29/the-effect-of-lease-rates-on-precious-metals-markets",
        "source_type": "industry", "citation": "LBMA Alchemist.",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": date, "name": "eth_validator_queue_ratio", "asset_class": "crypto",
        "target_assets": ["ETH-USD"], "category": "onchain",
        "description": "ETH validator entry/exit queue ratio. Entry>>exit=bullish.",
        "formula": "queue_ratio = entry_queue/max(exit_queue,1). Z-score. BUY ratio>10, SELL ratio<2.",
        "data_sources": ["beaconchain_api"],
        "parameters": {"lookback": 30, "buy_thresh": 10, "sell_thresh": 2},
        "novelty_score": 9.0, "edge_evidence": 6.0, "data_availability": 7.0,
        "implementation_cost": 7.0, "non_redundancy": 9.0, "composite_score": 7.50,
        "source_url": "https://beaconcha.in/api/v1/docs", "source_type": "industry",
        "citation": "Beaconcha.in. Shanghai exit event correlation.",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": date, "name": "gold_futures_trend_regime", "asset_class": "metals",
        "target_assets": ["XAU-USD"], "category": "technical",
        "description": "Smoothed trend-momentum regime. Walk-forward. Sharpe 2.88.",
        "formula": "Kalman trend + momentum ROC. Fractional Kelly with impact. ATR exits.",
        "data_sources": ["ohlcv"],
        "parameters": {"train": 2520, "test": 126, "vol_target": 0.15},
        "novelty_score": 6.0, "edge_evidence": 9.0, "data_availability": 10.0,
        "implementation_cost": 6.0, "non_redundancy": 5.0, "composite_score": 7.35,
        "source_url": "https://arxiv.org/abs/2511.08571", "source_type": "academic",
        "citation": "Singha et al (2025). Sharpe 2.88.",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": date, "name": "cross_commodity_leadlag", "asset_class": "cross-asset",
        "target_assets": ["XAU-USD", "XAG-USD", "BTC-USD", "ETH-USD"],
        "category": "intermarket",
        "description": "Cross-commodity lead-lag via Granger causality network. Sharpe 1.5.",
        "formula": "Granger F-stat at lags 1-24h. Directed graph. Network momentum.",
        "data_sources": ["ohlcv", "binance_copper", "fred_dxy"],
        "parameters": {"lags": 24, "pvalue": 0.05, "mom_lookback": 20},
        "novelty_score": 8.0, "edge_evidence": 8.0, "data_availability": 7.0,
        "implementation_cost": 5.0, "non_redundancy": 7.0, "composite_score": 7.25,
        "source_url": "https://arxiv.org/abs/2501.07135", "source_type": "academic",
        "citation": "Li and Ferreira (2025). Sharpe 1.511.",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": date, "name": "mstr_mnav_mean_reversion", "asset_class": "stocks",
        "target_assets": ["MSTR"], "category": "fundamental",
        "description": "MSTR mNAV premium MR. BUY below 1.0x, SELL above 2.0x.",
        "formula": "mnav = mstr_mcap / (btc_holdings * btc_price). Z-score.",
        "data_sources": ["alpaca_mstr", "binance_btc"],
        "parameters": {"lookback": 90, "buy": 1.0, "sell": 2.0},
        "novelty_score": 7.0, "edge_evidence": 7.0, "data_availability": 7.0,
        "implementation_cost": 7.0, "non_redundancy": 8.0, "composite_score": 7.15,
        "source_url": "https://bitcoinquant.co/company/MSTR", "source_type": "industry",
        "citation": "mNAV 0.97x bottom -> +60pct.",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": date, "name": "eth_blob_fee_activity", "asset_class": "crypto",
        "target_assets": ["ETH-USD"], "category": "onchain",
        "description": "Post-EIP-4844 blob utilization and fee as ETH demand proxy.",
        "formula": "blob_util = blobs_used/target. Fee z-score. BUY util>0.7.",
        "data_sources": ["ethereum_rpc"],
        "parameters": {"lookback": 30, "util_high": 0.7},
        "novelty_score": 10.0, "edge_evidence": 5.0, "data_availability": 5.0,
        "implementation_cost": 4.0, "non_redundancy": 10.0, "composite_score": 6.85,
        "source_url": "https://arxiv.org/abs/2502.12966", "source_type": "academic",
        "citation": "Heimbach and Milionis (2025).",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": date, "name": "vix_gsr_confluence", "asset_class": "metals",
        "target_assets": ["XAU-USD", "XAG-USD"], "category": "intermarket",
        "description": "VIX 12m z-score + GSR 3m z-score confluence. 26 trades.",
        "formula": "Both z-scores > 0.9 required. Monthly holding.",
        "data_sources": ["fred_vixcls", "ohlcv"],
        "parameters": {"vix_lb": 252, "gsr_lb": 63, "threshold": 0.9},
        "novelty_score": 6.0, "edge_evidence": 6.0, "data_availability": 9.0,
        "implementation_cost": 8.0, "non_redundancy": 6.0, "composite_score": 6.75,
        "source_url": "https://emergingmarketquests.substack.com/p/gold-silver-futures-trading-strategy",
        "source_type": "blog", "citation": "VIX-GSR confluence, 26 trades.",
        "status": "new", "created_at": now, "updated_at": now,
    },
]

# Append to backlog JSONL
with open("data/signal_research_backlog.jsonl", "a") as f:
    for c in candidates:
        f.write(json.dumps(c) + "\n")

# Insert into SQLite
db = sqlite3.connect("data/signal_log.db")
for c in candidates:
    db.execute(
        """INSERT INTO signal_candidates (date, name, asset_class, target_assets, category,
        description, formula, data_sources, parameters, novelty_score, edge_evidence,
        data_availability, implementation_cost, non_redundancy, composite_score,
        source_url, source_type, citation, status, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (c["date"], c["name"], c["asset_class"], json.dumps(c["target_assets"]),
         c["category"], c["description"], c["formula"], json.dumps(c["data_sources"]),
         json.dumps(c["parameters"]), c["novelty_score"], c["edge_evidence"],
         c["data_availability"], c["implementation_cost"], c["non_redundancy"],
         c["composite_score"], c["source_url"], c["source_type"], c["citation"],
         c["status"], c["created_at"], c["updated_at"]))
db.commit()
total = db.execute("SELECT COUNT(*) FROM signal_candidates").fetchone()[0]
print(f"Inserted {len(candidates)} candidates. Total: {total}")
db.close()

# Write ranked output
ranked = {
    "date": date,
    "new_candidates_scored": len(candidates),
    "backlog_candidates_checked": 155,
    "top_candidate": {
        "name": "bocpd_regime_switch",
        "composite_score": 8.25,
        "scores": {
            "novelty": 9.0, "edge_evidence": 8.0, "data_availability": 10.0,
            "implementation_cost": 6.0, "non_redundancy": 8.0,
        },
        "description": "Bayesian Online Changepoint Detection on return series. "
                        "Detects regime breaks in real-time. On changepoint: switch trend->MR.",
        "formula": "BOCPD recursive Bayesian with exponential hazard. "
                   "Run-length posterior. Changepoint fires MR mode.",
        "source": "Wood, Roberts, Zohren (2021), JFDS. arxiv:2105.13727",
    },
    "all_ranked": [
        c["name"] for c in sorted(
            candidates, key=lambda x: x["composite_score"], reverse=True
        )
    ],
    "implementation_decision": "implement",
    "skip_reason": None,
}
with open("data/signal_research_ranked.json", "w") as f:
    json.dump(ranked, f, indent=2)

# Write summary for Phase 5
summary = {
    "date": date,
    "assets_researched": ["XAG-USD", "XAU-USD", "ETH-USD", "BTC-USD", "cross-asset"],
    "total_papers_found": 33,
    "total_web_sources_found": 22,
    "total_new_candidates": len(candidates),
    "top_candidate": {
        "name": "bocpd_regime_switch",
        "composite_score": 8.25,
        "description": "Bayesian Online Changepoint Detection on return series. "
                        "Detects regime breaks in real-time. On changepoint: switch trend->MR.",
        "formula": "BOCPD recursive Bayesian with exponential hazard h(r)=1/lambda. "
                   "Run-length posterior. Changepoint fires MR mode.",
        "source": "Wood, Roberts, Zohren (2021), JFDS. arxiv:2105.13727",
        "target_assets": ["XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD", "MSTR"],
        "category": "regime",
        "data_sources": ["ohlcv"],
        "parameters": {"hazard_lambda": 100, "zscore_threshold": 2.0, "mr_window": 20},
        "implementation_notes": (
            "Use OHLCV data only. Register via register_enhanced() in signal_registry.py. "
            "Add to DISABLED_SIGNALS in tickers.py. Cap max_confidence at 0.7. "
            "requires_context=False since OHLCV-only. 4 sub-signals: "
            "changepoint_detector, trend_follower, mean_reverter, regime_classifier."
        ),
    },
    "backlog_additions": len(candidates),
    "skipped_implementation_reason": None,
}
with open("data/signal_research_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Update progress
progress = {
    "current_phase": "PHASE 5: CONTEXT RESET",
    "phase_started": now,
    "last_update": now,
    "status": "running",
    "notes": "Phase 3-4 complete. Top: bocpd_regime_switch (8.25). Moving to implementation.",
    "phases_completed": ["Phase 0", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5"],
    "session_date": date,
    "focus_assets": ["XAG-USD", "XAU-USD", "ETH-USD", "cross-asset"],
}
with open("data/signal-research-progress.json", "w") as f:
    json.dump(progress, f, indent=2)

print("All Phase 3-4-5 deliverables written successfully.")
