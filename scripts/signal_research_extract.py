"""Signal research Phase 3: Extract candidates and append to backlog."""
import json
import datetime
import sqlite3
from pathlib import Path

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
date = "2026-04-28"

candidates = [
    {
        "date": date,
        "name": "vol_ratio_regime",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        "category": "microstructure",
        "description": (
            "Garman-Klass to close-to-close volatility ratio as regime detector + directional signal. "
            "High GK/CC ratio means large intrabar swings but small close-to-close moves (ranging/mean-reverting). "
            "Low ratio means closes move consistently (trending). Sub-signals: gk_cc_z (z-scored ratio), "
            "variance_ratio (k-period VR test), efficiency_ratio (Kaufman ER). Directional: in ranging regime "
            "+ price below SMA -> BUY, above SMA -> SELL. In trending + direction up -> BUY, down -> SELL."
        ),
        "formula": (
            "GK_var = 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2. "
            "CC_var = (ln(C[t]/C[t-1]))^2. ratio = SMA(GK_var,20) / SMA(CC_var,20). "
            "VR = Var(5d_ret) / (5 * Var(1d_ret)). ER = abs(C[20]-C[0]) / sum(abs(C[i]-C[i-1])). "
            "Trending: ratio < 1.2 AND VR > 0.9 AND ER > 0.3. Ranging: ratio > 2.0 AND VR < 0.7 AND ER < 0.2."
        ),
        "data_sources": ["already_available:ohlcv"],
        "parameters": {"gk_lookback": 20, "vr_lookback": 60, "vr_k": 5, "er_lookback": 20, "sma_lookback": 20},
        "novelty_score": 8.5,
        "edge_evidence": 7.0,
        "data_availability": 10.0,
        "implementation_cost": 8.0,
        "non_redundancy": 9.0,
        "source_url": "https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/",
        "source_type": "academic+blog",
        "citation": "Garman-Klass 1980; Lo-MacKinlay 1988 (VR); Kaufman 1995 (ER)",
        "status": "new",
        "created_at": now,
        "updated_at": now,
    },
    {
        "date": date,
        "name": "cubic_trend_persistence",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        "category": "technical",
        "description": (
            "Cubic polynomial trend model. R(t+1) = a + b*phi(t) + c*phi(t)^3 where phi(t) is "
            "exponentially-weighted trend strength. Captures weak trend persistence (positive b) and "
            "strong trend exhaustion/reversion (negative c). Parameters universal across assets for >= 1h."
        ),
        "formula": (
            "phi(t) = sum(w(n) * (R(t-n) - mu) / sigma). w(n) = exp(-n/T) / sum(exp(-k/T)). "
            "Fit cubic on rolling window. BUY when phi > 0 AND phi < 1.5. SELL when phi > 2.0 (reversion)."
        ),
        "data_sources": ["already_available:ohlcv"],
        "parameters": {"T_lookback": 60, "fit_window": 252, "persistence_threshold": 1.5, "reversion_threshold": 2.0},
        "novelty_score": 9.0,
        "edge_evidence": 7.5,
        "data_availability": 10.0,
        "implementation_cost": 6.0,
        "non_redundancy": 8.0,
        "source_url": "https://arxiv.org/abs/2501.16772",
        "source_type": "academic",
        "citation": "Trends and Reversion in Financial Markets, arxiv:2501.16772, 2025",
        "status": "new",
        "created_at": now,
        "updated_at": now,
    },
    {
        "date": date,
        "name": "residual_pair_reversion",
        "asset_class": "cross-asset",
        "target_assets": ["ETH-USD", "XAG-USD", "XAU-USD"],
        "category": "intermarket",
        "description": (
            "Pairs-trade residual mean reversion. Regress correlated asset on driver (ETH~BTC, XAG~XAU) "
            "over 180d rolling window. Z-score residual. Trade mean reversion when z exceeds threshold. "
            "Sharpe 2.3 reported post-2021 for ETH-BTC. Portable to XAG-on-XAU."
        ),
        "formula": (
            "beta = OLS(ETH_ret ~ BTC_ret, window=180d). residual = ETH_ret - beta*BTC_ret. "
            "z = (residual - SMA(residual,60)) / std(residual,60). BUY when z < -2.0. SELL when z > 2.0."
        ),
        "data_sources": ["already_available:ohlcv_multi_asset"],
        "parameters": {"ols_window": 180, "z_lookback": 60, "z_threshold": 2.0},
        "novelty_score": 7.0,
        "edge_evidence": 8.5,
        "data_availability": 10.0,
        "implementation_cost": 8.0,
        "non_redundancy": 8.5,
        "source_url": "https://quantifiedstrategies.com",
        "source_type": "blog+research",
        "citation": "BTC-Neutral Residual Reversion, various quant blogs, 2025-2026",
        "status": "new",
        "created_at": now,
        "updated_at": now,
    },
    {
        "date": date,
        "name": "vol_rank_gate",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        "category": "technical",
        "description": (
            "Realized volatility percentile rank as trade gate. Reject trades when vol rank > 70th "
            "percentile. Backtested: profit factor 1.64 -> 2.52, drawdown 12.9% -> 8.7%. "
            "Directional: rv_rank dropping + uptrend -> BUY. rv_rank rising + downtrend -> SELL."
        ),
        "formula": (
            "rv = std(returns, 20). rv_rank = percentile_rank(rv, lookback=252). "
            "Gate: rv_rank > 70 -> force HOLD. rv_rank < 30 -> boost 1.2x. "
            "Directional: BUY when rv_rank dropping AND uptrend. SELL when rv_rank rising AND downtrend."
        ),
        "data_sources": ["already_available:ohlcv"],
        "parameters": {"rv_window": 20, "rank_lookback": 252, "high_vol_gate": 70, "low_vol_boost": 30},
        "novelty_score": 6.0,
        "edge_evidence": 8.0,
        "data_availability": 10.0,
        "implementation_cost": 9.0,
        "non_redundancy": 7.0,
        "source_url": "https://quantifiedstrategies.com",
        "source_type": "blog+backtest",
        "citation": "QuantifiedStrategies vol rank filter, 2025",
        "status": "new",
        "created_at": now,
        "updated_at": now,
    },
    {
        "date": date,
        "name": "trend_stability_filter",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        "category": "ensemble",
        "description": (
            "Persistence-based trend quality filter. 3 diagnostics: weight stddev, autocorrelation, "
            "max consecutive change. >= 2 of 3 met = stable trend. Sharpe 0.86 vs 0.79. "
            "78% improvement in weak-trending periods."
        ),
        "formula": (
            "sd < pct(40). ac > pct(60). mc < pct(40). stable = sum([sd,ac,mc]) >= 2. "
            "If stable: EMA-smoothed weights. Else: equal weight fallback."
        ),
        "data_sources": ["already_available:ohlcv"],
        "parameters": {"rolling_window": 60, "stability_window": 20, "alpha": 0.3},
        "novelty_score": 8.0,
        "edge_evidence": 8.0,
        "data_availability": 9.0,
        "implementation_cost": 5.0,
        "non_redundancy": 7.0,
        "source_url": "https://arxiv.org/abs/2510.23150",
        "source_type": "academic",
        "citation": "Revisiting Trend Premia, arxiv:2510.23150, 2025",
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
        1,
    )

# Check existing backlog
backlog_path = Path("data/signal_research_backlog.jsonl")
existing_names = set()
if backlog_path.exists():
    for line in backlog_path.read_text().strip().split("\n"):
        if line.strip():
            entry = json.loads(line)
            existing_names.add(entry["name"])

new_count = 0
with open(backlog_path, "a") as f:
    for c in candidates:
        if c["name"] not in existing_names:
            f.write(json.dumps(c) + "\n")
            new_count += 1
            print(f"  Added: {c['name']} (score={c['composite_score']})")
        else:
            print(f"  SKIP (dup): {c['name']}")

print(f"\nAdded {new_count} new candidates")

# Insert into SQLite
db = sqlite3.connect("data/signal_log.db")
db.execute(
    """CREATE TABLE IF NOT EXISTS signal_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL, name TEXT NOT NULL, asset_class TEXT NOT NULL,
    target_assets TEXT, category TEXT NOT NULL, description TEXT NOT NULL,
    formula TEXT, data_sources TEXT, parameters TEXT,
    novelty_score REAL, edge_evidence REAL, data_availability REAL,
    implementation_cost REAL, non_redundancy REAL, composite_score REAL,
    source_url TEXT, source_type TEXT, citation TEXT,
    status TEXT DEFAULT 'new', backtest_sharpe REAL, backtest_winrate REAL,
    backtest_notes TEXT, implemented_module TEXT, implemented_date TEXT,
    rejection_reason TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
)"""
)

for c in candidates:
    if c["name"] not in existing_names:
        db.execute(
            """INSERT INTO signal_candidates
            (date,name,asset_class,target_assets,category,description,formula,
             data_sources,parameters,novelty_score,edge_evidence,data_availability,
             implementation_cost,non_redundancy,composite_score,source_url,source_type,
             citation,status,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                c["date"], c["name"], c["asset_class"], json.dumps(c["target_assets"]),
                c["category"], c["description"], c["formula"],
                json.dumps(c["data_sources"]), json.dumps(c["parameters"]),
                c["novelty_score"], c["edge_evidence"], c["data_availability"],
                c["implementation_cost"], c["non_redundancy"], c["composite_score"],
                c["source_url"], c["source_type"], c["citation"], c["status"],
                c["created_at"], c["updated_at"],
            ),
        )
db.commit()
db.close()

print("\n=== New candidates ranked ===")
for c in sorted(candidates, key=lambda x: x["composite_score"], reverse=True):
    print(
        f"  {c['name']:35s} score={c['composite_score']}  "
        f"N={c['novelty_score']} E={c['edge_evidence']} "
        f"D={c['data_availability']} I={c['implementation_cost']} R={c['non_redundancy']}"
    )
