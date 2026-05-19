"""Write Phase 1-5 research outputs for signal research session 2026-04-26."""
import json
import datetime
import pathlib

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
today = "2026-04-26"

# Phase 1-2: Write papers and web sources
papers = {
    "date": today,
    "queries_run": [
        "intermarket cross asset momentum correlation trading (Semantic Scholar)",
        "precious metals trading signal prediction (arXiv q-fin)",
        "cryptocurrency order flow liquidity alpha (arXiv q-fin)",
        "gold silver ratio trading strategy signal (SSRN)",
        "cryptocurrency on-chain metrics trading alpha (SSRN)",
        "cross-asset intermarket momentum signal strategy (SSRN)",
    ],
    "papers_found": [
        {
            "title": "Financial Multiplex Network Model: Volume-Volatility Cross-Correlation",
            "authors": "Various",
            "year": 2025,
            "source": "semantic_scholar",
            "url": "https://www.semanticscholar.org/paper/1a529287eec87b3d71f1c50ee873f121a0a6816c",
            "citation_count": 2,
            "abstract_summary": "Three-layer networks predict future Sharpe ratio better than single-layer.",
            "signal_ideas": ["multiplex_network_sharpe_predictor"],
            "relevance": "medium",
            "assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        },
        {
            "title": "Forecast-to-Fill: Gold Futures Alpha Generation",
            "authors": "Singha, Aguilera-Toste, Lahiri",
            "year": 2025,
            "source": "arxiv",
            "url": "arXiv q-fin 2025",
            "citation_count": 0,
            "abstract_summary": "Trend+momentum for gold. Sharpe 2.88, 43% annual. Kelly sizing.",
            "signal_ideas": ["gold_trend_momentum_kelly"],
            "relevance": "high",
            "assets": ["XAU-USD"],
        },
        {
            "title": "Gold Silver Pair Trading - Mean Reversion with ML",
            "authors": "Mittal, Mittal",
            "year": 2025,
            "source": "ssrn",
            "url": "https://ssrn.com/abstract=5710242",
            "citation_count": 0,
            "abstract_summary": "ML-filtered gold-silver spread mean reversion. GB + SVM regime filter.",
            "signal_ideas": ["gsr_ml_filtered_mean_reversion"],
            "relevance": "high",
            "assets": ["XAU-USD", "XAG-USD"],
        },
        {
            "title": "Bitcoin Return Predictability: Crypto-Native Variables",
            "authors": "Palazzi, Raimundo, Klotzle",
            "year": 2026,
            "source": "ssrn",
            "url": "https://ssrn.com/abstract=6199098",
            "citation_count": 0,
            "abstract_summary": "MVRV, funding, OI predict BTC across regimes. Crypto-native > macro.",
            "signal_ideas": ["btc_crypto_native_factor_model"],
            "relevance": "high",
            "assets": ["BTC-USD"],
        },
        {
            "title": "Catching Crypto Trends: Tactical Approach",
            "authors": "Zarattini, Pagani, Barbon",
            "year": 2025,
            "source": "ssrn",
            "url": "https://ssrn.com/abstract=5209907",
            "citation_count": 0,
            "abstract_summary": "Trend-following top 20 coins. Sharpe >1.5, 10.8% alpha vs BTC.",
            "signal_ideas": ["crypto_rotational_momentum"],
            "relevance": "medium",
            "assets": ["BTC-USD", "ETH-USD"],
        },
        {
            "title": "Explainable Patterns in Cryptocurrency Microstructure",
            "authors": "Bieganowski, Slepaczuk",
            "year": 2025,
            "source": "arxiv",
            "url": "https://arxiv.org/abs/2602.00776",
            "citation_count": 0,
            "abstract_summary": "Cross-asset LOB patterns. OFI, spread, adverse selection. SHAP.",
            "signal_ideas": ["explainable_lob_microstructure"],
            "relevance": "medium",
            "assets": ["BTC-USD", "ETH-USD"],
        },
    ],
}

web = {
    "date": today,
    "queries_run": [
        "SSR bitcoin prediction", "hash ribbon formula", "ETH validator queue",
        "gold GVZ skew", "futures basis signal", "gold central bank buying",
        "alphaarchitect cross-asset", "defillama API", "bgeometrics hashrate",
        "funding rate arbitrage sharpe",
    ],
    "sources_found": [
        {"title": "Hash Ribbons (VanEck + CoinDesk)", "url": "coindesk.com", "source_type": "industry", "signal_ideas": ["hash_ribbons_btc"], "relevance": "high", "assets": ["BTC-USD"]},
        {"title": "Stablecoin Supply Ratio (CryptoQuant)", "url": "cryptoquant.com", "source_type": "industry", "signal_ideas": ["stablecoin_supply_ratio"], "relevance": "high", "assets": ["BTC-USD"]},
        {"title": "ETH Validator Queue Balance", "url": "beaconcha.in", "source_type": "industry", "signal_ideas": ["eth_validator_queue_balance"], "relevance": "medium", "assets": ["ETH-USD"]},
        {"title": "DefiLlama Stablecoins API", "url": "api.llama.fi", "source_type": "industry", "signal_ideas": ["stablecoin_supply_ratio"], "relevance": "high", "assets": ["BTC-USD"]},
        {"title": "BTC Miner Capitulation Feb 2026", "url": "coindesk.com", "source_type": "industry", "signal_ideas": ["hash_ribbons_btc"], "relevance": "high", "assets": ["BTC-USD"]},
        {"title": "Cross-Asset Base Pairs (Alpha Architect)", "url": "alphaarchitect.com", "source_type": "blog", "signal_ideas": ["cross_asset_base_pair"], "relevance": "medium", "assets": ["cross-asset"]},
        {"title": "Funding Rate CEX vs DEX (ScienceDirect)", "url": "sciencedirect.com", "source_type": "academic", "signal_ideas": ["funding_rate_cex_dex"], "relevance": "medium", "assets": ["BTC-USD"]},
        {"title": "Gold Central Bank Buying (WGC)", "url": "gold.org", "source_type": "industry", "signal_ideas": ["central_bank_gold"], "relevance": "low", "assets": ["XAU-USD"]},
        {"title": "CME CVOL Skew", "url": "cmegroup.com", "source_type": "industry", "signal_ideas": ["metals_cvol_skew"], "relevance": "medium", "assets": ["XAU-USD", "XAG-USD"]},
        {"title": "BGeometrics hashrate endpoint", "url": "bitcoin-data.com", "source_type": "industry", "signal_ideas": ["hash_ribbons_btc"], "relevance": "high", "assets": ["BTC-USD"]},
    ],
}

with open("data/signal_research_papers.json", "w") as f:
    json.dump(papers, f, indent=2)
with open("data/signal_research_web.json", "w") as f:
    json.dump(web, f, indent=2)
print(f"Papers: {len(papers['papers_found'])} | Web: {len(web['sources_found'])}")

# Phase 3: Add new candidates to backlog
new_candidates = [
    {
        "date": today, "name": "eth_validator_queue_balance", "asset_class": "crypto",
        "target_assets": ["ETH-USD"], "category": "on-chain",
        "description": "ETH validator entry/exit queue imbalance as supply signal.",
        "formula": "queue_ratio = entry_queue_eth / max(exit_queue_eth, 1). BUY > 10, SELL < 0.1.",
        "data_sources": ["Beaconcha.in free API"],
        "parameters": {"buy_threshold": 10, "sell_threshold": 0.1},
        "novelty_score": 9.0, "edge_evidence": 5.0, "data_availability": 6.0,
        "implementation_cost": 6.0, "non_redundancy": 9.0, "composite_score": 6.9,
        "source_url": "https://beaconcha.in/validators/queues",
        "source_type": "industry", "citation": "Beaconcha.in + Yahoo Finance",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": today, "name": "gsr_ml_filtered_mean_reversion", "asset_class": "metals",
        "target_assets": ["XAU-USD", "XAG-USD"], "category": "intermarket",
        "description": "G/S ratio mean reversion with ML regime filter.",
        "formula": "z-scored GSR + GB classifier for stable regime.",
        "data_sources": ["XAU/XAG prices (existing)"],
        "parameters": {"z_threshold": 2.0, "sma_period": 60},
        "novelty_score": 7.0, "edge_evidence": 7.0, "data_availability": 8.0,
        "implementation_cost": 5.0, "non_redundancy": 6.0, "composite_score": 6.65,
        "source_url": "https://ssrn.com/abstract=5710242",
        "source_type": "academic", "citation": "Mittal & Mittal 2025, SSRN",
        "status": "new", "created_at": now, "updated_at": now,
    },
    {
        "date": today, "name": "crypto_native_factor_model", "asset_class": "crypto",
        "target_assets": ["BTC-USD"], "category": "on-chain",
        "description": "Multi-factor: MVRV + funding + OI. Rolling regression.",
        "formula": "Rolling 60d regression on [mvrv_z, funding_z, oi_pct].",
        "data_sources": ["BGeometrics (existing)", "Binance (existing)"],
        "parameters": {"lookback": 60},
        "novelty_score": 5.0, "edge_evidence": 7.0, "data_availability": 9.0,
        "implementation_cost": 7.0, "non_redundancy": 4.0, "composite_score": 6.25,
        "source_url": "https://ssrn.com/abstract=6199098",
        "source_type": "academic", "citation": "Palazzi et al. 2026",
        "status": "new", "created_at": now, "updated_at": now,
    },
]

with open("data/signal_research_backlog.jsonl", "a") as f:
    for c in new_candidates:
        f.write(json.dumps(c) + "\n")
print(f"Added {len(new_candidates)} new candidates to backlog")

# Phase 4: Ranked
ranked = {
    "date": today, "new_candidates_scored": 3, "backlog_candidates_checked": 130,
    "top_candidate": {
        "name": "hash_ribbons_btc", "composite_score": 8.27,
        "scores": {"novelty": 8.5, "edge_evidence": 9.0, "data_availability": 7.0, "implementation_cost": 7.0, "non_redundancy": 9.0},
        "description": "30d/60d SMA hashrate crossover. 89% win rate, 9 signals since 2011.",
        "formula": "BUY when SMA(hashrate,30) crosses above SMA(hashrate,60) AND SMA(close,10) > SMA(close,20).",
        "source": "Capriole Investments; VanEck; CoinDesk. Backlog 2026-04-11.",
    },
    "all_ranked": [
        "hash_ribbons_btc (8.27)", "multi_llm_disagreement (8.15)",
        "amihud_illiquidity_regime (8.15)", "eth_validator_queue_balance (6.9)",
    ],
    "implementation_decision": "implement", "skip_reason": None,
}
with open("data/signal_research_ranked.json", "w") as f:
    json.dump(ranked, f, indent=2)

# Phase 5: Summary
summary = {
    "date": today,
    "assets_researched": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "cross-asset"],
    "total_papers_found": 6, "total_web_sources_found": 10, "total_new_candidates": 3,
    "top_candidate": {
        "name": "hash_ribbons_btc",
        "description": "BTC miner capitulation detector. 30d/60d SMA hashrate crossover. 89% win rate.",
        "formula": "BUY when SMA(hashrate,30) crosses above SMA(hashrate,60) AND SMA(close,10) > SMA(close,20).",
        "data_sources": ["blockchain.info free API", "BTC OHLCV (existing)"],
        "parameters": {"hash_fast": 30, "hash_slow": 60, "price_fast": 10, "price_slow": 20},
        "composite_score": 8.27,
        "source": "Capriole Investments; VanEck; CoinDesk",
        "target_assets": ["BTC-USD"], "category": "on-chain",
        "implementation_notes": "blockchain.info API for hashrate. 24h cache. DISABLED_SIGNALS shadow mode.",
    },
    "backlog_additions": 3, "skipped_implementation_reason": None,
}
with open("data/signal_research_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Update progress
progress_file = pathlib.Path("data/signal-research-progress.json")
progress = json.loads(progress_file.read_text())
progress.update({
    "current_phase": "PHASE 6: IMPLEMENT",
    "phase_started": now, "last_update": now, "status": "running",
    "notes": "Implementing hash_ribbons_btc (8.27). blockchain.info for hashrate.",
    "phases_completed": [
        "PHASE 0: BASELINE", "PHASE 1: ACADEMIC SEARCH", "PHASE 2: WEB RESEARCH",
        "PHASE 3: EXTRACTION", "PHASE 4: SCORING", "PHASE 5: CONTEXT RESET",
    ],
})
progress_file.write_text(json.dumps(progress, indent=2))
print("All Phase 1-5 outputs written. Progress: Phase 6")
