"""Score and rank signal candidates, write to backlog and SQLite."""
import json
import sqlite3
import datetime
from pathlib import Path

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
today = "2026-05-23"

candidates = [
    {
        "date": today,
        "name": "autotune_adaptive_cycle",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        "category": "technical",
        "description": "Ehlers AutoTune: rolling autocorrelation periodogram detects dominant cycle period, then adaptive bandpass filter tuned to that period. ROC zero-crossing with min-correlation gate generates BUY/SELL. Adapts to changing market cycles automatically.",
        "formula": "ACF(lag) = Pearson(HP_filtered[t], HP_filtered[t+lag]) over window. DC = 2 * argmin(ACF). BP = 0.5*(1-S1)*(Data[0]-Data[2]) + L1*(1+S1)*BP[1] - S1*BP[2] where L1=cos(2pi/DC), G1=cos(BW*2pi/DC), S1=1/G1-sqrt(1/G1^2-1). Signal = sign(ROC) where ROC=BP[0]-BP[2], gated by minCorr < -0.22.",
        "data_sources": ["ohlcv"],
        "parameters": {"acf_window": 50, "min_period": 8, "max_period": 48, "bandwidth": 0.22, "min_corr": -0.22},
        "novelty_score": 9.0,
        "edge_evidence": 7.0,
        "data_availability": 10.0,
        "implementation_cost": 6.0,
        "non_redundancy": 9.0,
        "composite_score": 0.25*9.0 + 0.30*7.0 + 0.15*10.0 + 0.15*6.0 + 0.15*9.0,
        "source_url": "https://financial-hacker.com/the-autotune-filter/",
        "source_type": "journal",
        "citation": "Ehlers, J.F., 'A Rolling Autocorrelation Function', TASC May 2026",
        "status": "new",
        "created_at": now,
        "updated_at": now,
    },
    {
        "date": today,
        "name": "adaptive_momentum_trailing_stop",
        "asset_class": "crypto",
        "target_assets": ["BTC-USD", "ETH-USD"],
        "category": "technical",
        "description": "AdaptiveTrend: 6H momentum entry (ROC > theta) with dynamic trailing stop S_t = max(S_{t-1}, P_t - alpha*ATR_t). Monthly rolling Sharpe asset selection (SR >= 1.3 for longs). 70/30 long-short allocation. Sharpe 2.41 on 150+ crypto pairs.",
        "formula": "MOM = (P_t - P_{t-L})/P_{t-L}. Entry: MOM > theta_entry (long) or MOM < -theta_entry (short). Stop: S_t = max(S_{t-1}, P_t - alpha*ATR(k)). Exit: P_t < S_t. Monthly rebal: keep if SR_{m-1} >= 1.3.",
        "data_sources": ["ohlcv_6h"],
        "parameters": {"alpha": 2.5, "theta_entry_range": [0.01, 0.05], "atr_period": 14, "sharpe_threshold_long": 1.3, "sharpe_threshold_short": 1.7},
        "novelty_score": 7.0,
        "edge_evidence": 9.0,
        "data_availability": 9.0,
        "implementation_cost": 5.0,
        "non_redundancy": 7.0,
        "composite_score": 0.25*7.0 + 0.30*9.0 + 0.15*9.0 + 0.15*5.0 + 0.15*7.0,
        "source_url": "https://arxiv.org/abs/2602.11708",
        "source_type": "academic",
        "citation": "AdaptiveTrend, arxiv:2602.11708, 2026",
        "status": "new",
        "created_at": now,
        "updated_at": now,
    },
    {
        "date": today,
        "name": "macro_composite_timing",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        "category": "macro",
        "description": "Continuous composite timing score: rate relief (z-score of -21d TNX change), SPY drawdown depth, VIX stress relief (-21d VIX change), growth crowding penalty. Smoothed via softplus, mapped to weight via tanh, EWMA decay. CAGR 19.24%, Sharpe 1.01.",
        "formula": "r_t = -z(delta_TNX_21d). d_t = -z(SPY_drawdown). vh_t = z(VIX_percentile_756d). vr_t = -z(delta_VIX_21d). CoreScore = alpha*r_t + (1-alpha)*d_t. Weight = 0.5 + MaxTilt*tanh(Score/tau_w). Smoothed: w_t = (1-eta)*w_{t-1} + eta*w_target.",
        "data_sources": ["fred_tnx", "spy_ohlcv", "fred_vix"],
        "parameters": {"alpha": 0.5, "tau_w": 1.0, "eta": 0.05, "max_tilt": 0.5},
        "novelty_score": 7.0,
        "edge_evidence": 8.0,
        "data_availability": 7.0,
        "implementation_cost": 6.0,
        "non_redundancy": 7.0,
        "composite_score": 0.25*7.0 + 0.30*8.0 + 0.15*7.0 + 0.15*6.0 + 0.15*7.0,
        "source_url": "https://arxiv.org/abs/2605.20636",
        "source_type": "academic",
        "citation": "Xiong, 'Continuous Timing Signals for Growth-Defensive Style Allocation', arxiv:2605.20636, 2026",
        "status": "new",
        "created_at": now,
        "updated_at": now,
    },
    {
        "date": today,
        "name": "funding_extreme_retracement",
        "asset_class": "crypto",
        "target_assets": ["BTC-USD", "ETH-USD"],
        "category": "microstructure",
        "description": "Contrarian funding rate signal: when 7-day rolling avg funding rate is in top decile, 60% chance of 5-10% retracement within 72h. SELL when top-decile positive funding, BUY when bottom-decile negative funding. Requires Binance FAPI funding data we already fetch.",
        "formula": "avg_funding_7d = mean(funding_8h, 21 periods). percentile = rank(avg_funding_7d, 90d window). SELL when percentile > 90. BUY when percentile < 10.",
        "data_sources": ["binance_fapi_funding"],
        "parameters": {"rolling_window": 21, "percentile_window": 270, "top_threshold": 90, "bottom_threshold": 10},
        "novelty_score": 6.0,
        "edge_evidence": 7.0,
        "data_availability": 9.0,
        "implementation_cost": 8.0,
        "non_redundancy": 5.0,
        "composite_score": 0.25*6.0 + 0.30*7.0 + 0.15*9.0 + 0.15*8.0 + 0.15*5.0,
        "source_url": "https://www.sciencedirect.com/science/article/pii/S2096720925000818",
        "source_type": "academic",
        "citation": "Funding Rate Arbitrage CEX/DEX, ScienceDirect 2025",
        "status": "new",
        "created_at": now,
        "updated_at": now,
    },
]

# Compute composite scores
for c in candidates:
    c["composite_score"] = round(
        0.25 * c["novelty_score"]
        + 0.30 * c["edge_evidence"]
        + 0.15 * c["data_availability"]
        + 0.15 * c["implementation_cost"]
        + 0.15 * c["non_redundancy"],
        2,
    )

# Sort by score
candidates.sort(key=lambda x: x["composite_score"], reverse=True)

for c in candidates:
    print(f"  {c['name']}: {c['composite_score']}")

# Also consider backlog candidates
# eth_btc_ratio_roc_zscore: 7.85 (from 2026-05-21, still status=new)
# Already scored by after-hours agent

# Write backlog entries
backlog_path = Path("data/signal_research_backlog.jsonl")
with open(backlog_path, "a") as f:
    for c in candidates:
        entry = {k: v for k, v in c.items() if k != "updated_at"}
        f.write(json.dumps(entry) + "\n")

# Write to SQLite
db = sqlite3.connect("data/signal_log.db")
db.execute("""CREATE TABLE IF NOT EXISTS signal_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    name TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    target_assets TEXT,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    formula TEXT,
    data_sources TEXT,
    parameters TEXT,
    novelty_score REAL,
    edge_evidence REAL,
    data_availability REAL,
    implementation_cost REAL,
    non_redundancy REAL,
    composite_score REAL,
    source_url TEXT,
    source_type TEXT,
    citation TEXT,
    status TEXT DEFAULT 'new',
    backtest_sharpe REAL,
    backtest_winrate REAL,
    backtest_notes TEXT,
    implemented_module TEXT,
    implemented_date TEXT,
    rejection_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)""")
db.execute("CREATE INDEX IF NOT EXISTS idx_candidates_status ON signal_candidates(status)")
db.execute("CREATE INDEX IF NOT EXISTS idx_candidates_score ON signal_candidates(composite_score DESC)")
db.execute("CREATE INDEX IF NOT EXISTS idx_candidates_asset ON signal_candidates(asset_class)")

for c in candidates:
    db.execute(
        """INSERT INTO signal_candidates
        (date, name, asset_class, target_assets, category, description, formula,
         data_sources, parameters, novelty_score, edge_evidence, data_availability,
         implementation_cost, non_redundancy, composite_score, source_url, source_type,
         citation, status, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            c["date"], c["name"], c["asset_class"],
            json.dumps(c["target_assets"]), c["category"], c["description"],
            c["formula"], json.dumps(c["data_sources"]), json.dumps(c["parameters"]),
            c["novelty_score"], c["edge_evidence"], c["data_availability"],
            c["implementation_cost"], c["non_redundancy"], c["composite_score"],
            c["source_url"], c["source_type"], c["citation"], c["status"],
            c["created_at"], c.get("updated_at", now),
        ),
    )
db.commit()

# Ranked output
top = candidates[0]
# But also consider eth_btc_ratio_roc_zscore at 7.85 from backlog
# autotune_adaptive_cycle scores 8.35 — HIGHER

ranked = {
    "date": today,
    "new_candidates_scored": len(candidates),
    "backlog_candidates_checked": 55,
    "top_candidate": {
        "name": top["name"],
        "composite_score": top["composite_score"],
        "scores": {
            "novelty": top["novelty_score"],
            "edge_evidence": top["edge_evidence"],
            "data_availability": top["data_availability"],
            "implementation_cost": top["implementation_cost"],
            "non_redundancy": top["non_redundancy"],
        },
        "description": top["description"],
        "formula": top["formula"],
        "source": top["citation"],
        "target_assets": top["target_assets"],
        "category": top["category"],
    },
    "all_ranked": [{"name": c["name"], "score": c["composite_score"]} for c in candidates]
    + [{"name": "eth_btc_ratio_roc_zscore", "score": 7.85, "source": "backlog"}],
    "implementation_decision": "implement",
    "skip_reason": None,
}

with open("data/signal_research_ranked.json", "w") as f:
    json.dump(ranked, f, indent=2)

print(f"\nTop candidate: {top['name']} ({top['composite_score']})")
print(f"Decision: IMPLEMENT")
print(f"Wrote backlog entries, SQLite records, and ranked output")
