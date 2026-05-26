"""Extract new signal candidates from research agent outputs, deduplicate, append to backlog."""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def slug(name):
    return name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("/", "_")

def main():
    # Load existing backlog names
    with open("data/signal_research_backlog.jsonl") as f:
        existing = [json.loads(l) for l in f if l.strip()]
    existing_names = set(e.get("name", "").lower().strip() for e in existing)

    # Load registered signal names
    with open("portfolio/tickers.py") as f:
        txt = f.read()
    registered = set(re.findall(r'"([a-z_]+)"', txt))
    all_known = existing_names | registered

    print(f"Existing backlog: {len(existing_names)}")
    print(f"Registered signals: {len(registered)}")
    print(f"Combined known: {len(all_known)}")

    new_candidates = []

    # From SSRN/industry synthesis
    with open("data/paper_search_ssrn_industry_2026-05-26.json") as f:
        ssrn = json.load(f)
    synth = ssrn.get("synthesis", {})
    for item in synth.get("highest_priority_new_signals", []):
        name = slug(item["name"])
        if name not in all_known:
            new_candidates.append({
                "name": name,
                "title": item["name"],
                "description": item["description"],
                "source": "ssrn_industry_2026-05-26",
                "priority": item.get("priority", 5),
                "assets": item.get("expected_assets", []),
                "effort": item.get("estimated_implementation_effort", "medium"),
                "rationale": item.get("rationale", ""),
                "data_available": item.get("data_available", True),
                "status": "new",
                "added_date": "2026-05-26",
            })

    # Manual extraction of novel signals from papers
    manual_ideas = [
        {
            "name": "nvt_signal_contrarian",
            "title": "NVT Signal Contrarian",
            "description": "Network Value to Transactions ratio. High NVT = overvalued (SELL), low = undervalued (BUY). Hybrid Transformer 61.25% accuracy, Sharpe 1.58.",
            "source": "Synergistic Alpha (2025), DOI:10.61996/economy.v3i2.103",
            "priority": 4,
            "assets": ["BTC-USD"],
            "effort": "medium",
            "backtest_metrics": {"accuracy": 0.6125, "sharpe": 1.58},
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "whale_transaction_monitor",
            "title": "Whale Transaction Monitor",
            "description": "Track large BTC transactions (>100 BTC). Short-term price impact significant per ARDL model (2026). Fades long-run.",
            "source": "arxiv:2602.08429",
            "priority": 5,
            "assets": ["BTC-USD"],
            "effort": "high",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "btc_dominance_rotation",
            "title": "BTC Dominance Rotation Signal",
            "description": "BTC dominance > 60% = institutional-led. Peak + reversal signals altcoin rotation (bullish ETH).",
            "source": "DOI:10.1080/23322039.2026.2625541",
            "priority": 4,
            "assets": ["BTC-USD", "ETH-USD"],
            "effort": "low",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "halving_cycle_positioning",
            "title": "BTC Halving Cycle Positioning",
            "description": "150-day pre/post halving windows show statistically significant BTC outperformance.",
            "source": "Sarkar (2025), SSRN:5395221",
            "priority": 6,
            "assets": ["BTC-USD"],
            "effort": "low",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "metals_copula_tail_dependence",
            "title": "Precious Metals Copula Tail Dependence",
            "description": "Copula-based dependency between gold/silver/platinum. Detects tail co-movement Pearson misses.",
            "source": "Macalli (2025), SSRN:5764942",
            "priority": 5,
            "assets": ["XAU-USD", "XAG-USD"],
            "effort": "high",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "sek_metals_efficiency_regime",
            "title": "SEK-Denominated Metals Efficiency Regime",
            "description": "Adaptive Market Hypothesis: SEK-denominated metals show different efficiency regimes than USD.",
            "source": "Rana et al (2025), SSRN:5225166",
            "priority": 6,
            "assets": ["XAU-USD", "XAG-USD"],
            "effort": "high",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "garch_chronos_vol_ensemble",
            "title": "GARCH + Chronos Volatility Ensemble",
            "description": "GARCH matches LSTM on MSE; Chronos outperforms MAE/QLIKE. Ensemble for metals vol forecasting.",
            "source": "Sieradzki & Kwiatek (2025), SSRN:5330706",
            "priority": 5,
            "assets": ["XAU-USD", "XAG-USD"],
            "effort": "medium",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "mfdfa_complexity_gate",
            "title": "Multifractal DFA Complexity Gate",
            "description": "MF-DFA and RCMSE complexity. Higher complexity in BTC than gold/FX. Volatility regime gate.",
            "source": "Masoudi et al (2025), arxiv:2507.23414",
            "priority": 6,
            "assets": ["BTC-USD", "XAU-USD"],
            "effort": "high",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "bayesian_gpr_tail_risk",
            "title": "Bayesian GPR Tail Risk Detector",
            "description": "Bayesian Generalized Pareto Regression with Cauchy prior for extreme loss forecasting.",
            "source": "Das (2025), arxiv:2506.17549",
            "priority": 6,
            "assets": ["BTC-USD", "XAU-USD", "XAG-USD"],
            "effort": "high",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "off_chain_demand_pressure",
            "title": "Off-Chain Demand Pressure Signal",
            "description": "ARDL: off-chain demand (exchange volume, derivatives OI) has long-run BTC price impact.",
            "source": "arxiv:2602.08429 (2026)",
            "priority": 5,
            "assets": ["BTC-USD"],
            "effort": "medium",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "comex_silver_registered_eligible",
            "title": "COMEX Silver Registered/Eligible Ratio",
            "description": "Declining ratio = physical squeeze risk. 1.1B oz drawn since 2019, 6th consecutive deficit.",
            "source": "Sprott (2025) + Silver Institute (2026)",
            "priority": 4,
            "assets": ["XAG-USD"],
            "effort": "medium",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "central_bank_gold_momentum",
            "title": "Central Bank Gold Purchase Momentum",
            "description": "Monthly CB net gold purchases. Deceleration while prices rise = bearish divergence.",
            "source": "WGC Q1 2026 + Amundi Research",
            "priority": 4,
            "assets": ["XAU-USD"],
            "effort": "medium",
            "status": "new",
            "added_date": "2026-05-26",
        },
        {
            "name": "cross_asset_bond_equity_momentum",
            "title": "Cross-Asset Bond-Equity Momentum Spillover",
            "description": "Past government bond returns predict commodity/equity returns. Time-series momentum across asset boundaries.",
            "source": "Kusakabe (2025), SSRN:5858542",
            "priority": 5,
            "assets": ["BTC-USD", "XAU-USD", "XAG-USD"],
            "effort": "medium",
            "status": "new",
            "added_date": "2026-05-26",
        },
    ]

    for c in manual_ideas:
        if c["name"] not in all_known:
            new_candidates.append(c)

    # Dedup within new_candidates
    seen = set()
    deduped = []
    for c in new_candidates:
        if c["name"] not in seen:
            seen.add(c["name"])
            deduped.append(c)

    print(f"\nNew candidates after dedup: {len(deduped)}")
    for c in deduped:
        print(f"  {c['name']} ({c.get('effort', '?')}): {c['assets']}")

    # Append to backlog
    with open("data/signal_research_backlog.jsonl", "a") as f:
        for c in deduped:
            f.write(json.dumps(c) + "\n")

    print(f"\nAppended {len(deduped)} new candidates to backlog")
    print(f"Total backlog now: {len(existing) + len(deduped)}")

if __name__ == "__main__":
    main()
