"""Phase 3-4: Extract signal candidates, score, and write to backlog + SQLite."""
import json
import datetime
import pathlib
import sqlite3

date_str = "2026-04-23"
now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

candidates = [
    {
        "date": date_str,
        "name": "realized_skewness_directional",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        "category": "technical",
        "description": (
            "Realized skewness of daily returns as directional signal. 3rd moment "
            "captures asymmetry: negative skew = fat left tail (overdue for mean "
            "reversion BUY), positive skew = fat right tail (momentum exhaustion "
            "SELL). 12-month lookback, z-scored. Sharpe 0.79, 8.01% annual on 27 "
            "commodity futures. Validated on crypto cross-section."
        ),
        "formula": (
            "skew = scipy.stats.skew(daily_returns[-lookback:]). "
            "z = (skew - SMA(skew, norm)) / StdDev(skew, norm). "
            "BUY z < -1.5. SELL z > 1.5. "
            "Sub: raw_skew, skew_z, skew_momentum, kurtosis_confirmation."
        ),
        "data_sources": ["already_available:ohlcv"],
        "parameters": {"lookback": 252, "norm_window": 60, "z_buy": -1.5, "z_sell": 1.5},
        "novelty_score": 9.0,
        "edge_evidence": 8.0,
        "data_availability": 10.0,
        "implementation_cost": 9.0,
        "non_redundancy": 9.0,
        "source_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2671165",
        "source_type": "academic",
        "citation": (
            "Fernandez-Perez et al (2018). Skewness of Commodity Futures Returns. "
            "JBF. Sharpe 0.79. Also ScienceDirect (2024) crypto skewness."
        ),
        "status": "new",
        "created_at": now_iso,
        "updated_at": now_iso,
    },
    {
        "date": date_str,
        "name": "amihud_illiquidity_regime",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        "category": "microstructure",
        "description": (
            "Amihud ILLIQ ratio as liquidity regime detector. "
            "ILLIQ = |return|/dollar_volume. Spike = thin market, breakouts are "
            "fakeouts. Falling ILLIQ = thick market, breakouts have conviction. "
            "Z-scored against 60d baseline."
        ),
        "formula": (
            "illiq = abs(pct_change(close)) / (close * volume). "
            "z = (illiq - SMA60) / StdDev60. "
            "BUY z < -1.0 AND uptrend. SELL z > 2.0. "
            "Sub: illiq_z, illiq_trend, volume_conf."
        ),
        "data_sources": ["already_available:ohlcv"],
        "parameters": {"lookback": 60, "z_buy": -1.0, "z_sell": 2.0},
        "novelty_score": 8.0,
        "edge_evidence": 7.0,
        "data_availability": 10.0,
        "implementation_cost": 9.0,
        "non_redundancy": 8.0,
        "source_url": "https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf",
        "source_type": "academic",
        "citation": "Amihud (2002). Illiquidity and Stock Returns. JFM. 120+ citations.",
        "status": "new",
        "created_at": now_iso,
        "updated_at": now_iso,
    },
    {
        "date": date_str,
        "name": "return_dispersion_sync",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"],
        "category": "regime",
        "description": (
            "Cross-asset return dispersion as market synchronization detector. "
            "StdDev of daily returns across 5 assets. Low = synchronization (risk). "
            "High = idiosyncratic (normal). Predicts future realized volatility."
        ),
        "formula": (
            "disp = std([ret_btc, ret_eth, ret_xau, ret_xag, ret_mstr]). "
            "z = (disp - SMA60) / StdDev60. SELL z < -1.5. BUY z > 1.0."
        ),
        "data_sources": ["already_available:ohlcv for all 5 assets"],
        "parameters": {"z_window": 60, "z_sell": -1.5, "z_buy": 1.0},
        "novelty_score": 8.0,
        "edge_evidence": 7.0,
        "data_availability": 10.0,
        "implementation_cost": 8.0,
        "non_redundancy": 7.0,
        "source_url": "https://onlinelibrary.wiley.com/doi/10.1002/for.2959",
        "source_type": "academic",
        "citation": "Niu (2023). Cross-sectional return dispersion and volatility. JoF. Sharpe 0.93.",
        "status": "new",
        "created_at": now_iso,
        "updated_at": now_iso,
    },
    {
        "date": date_str,
        "name": "matched_filter_ofi_norm",
        "asset_class": "cross-asset",
        "target_assets": ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD"],
        "category": "microstructure",
        "description": (
            "Matched filter normalization for existing OFI. Market-cap norm for "
            "institutional, volume norm for VWAP flows. 1.99x signal correlation."
        ),
        "formula": (
            "ofi_mc = raw_ofi / market_cap. ofi_tv = raw_ofi / volume. "
            "Pick highest-corr norm per asset."
        ),
        "data_sources": ["existing orderbook data + market cap"],
        "parameters": {"norm_type": "adaptive"},
        "novelty_score": 6.0,
        "edge_evidence": 8.0,
        "data_availability": 6.0,
        "implementation_cost": 5.0,
        "non_redundancy": 3.0,
        "source_url": "https://arxiv.org/abs/2512.18648",
        "source_type": "academic",
        "citation": "Kang (2025). Optimal Signal Extraction from Order Flow. arXiv 2512.18648.",
        "status": "new",
        "created_at": now_iso,
        "updated_at": now_iso,
    },
    {
        "date": date_str,
        "name": "global_liquidity_crypto",
        "asset_class": "crypto",
        "target_assets": ["BTC-USD", "ETH-USD"],
        "category": "macro",
        "description": (
            "Global M2 money supply + Fed balance sheet as crypto directional signal. "
            "1% liquidity shock = 2% BTC impact with 3-month lag."
        ),
        "formula": (
            "m2_mom = pct_change(M2, 30d). fed_mom = pct_change(fed_bs, 30d). "
            "BUY both > 0.005. SELL both < -0.005."
        ),
        "data_sources": ["FRED: M2SL, WALCL"],
        "parameters": {"lookback": 30, "threshold": 0.005},
        "novelty_score": 7.0,
        "edge_evidence": 6.0,
        "data_availability": 7.0,
        "implementation_cost": 7.0,
        "non_redundancy": 6.0,
        "source_url": "https://capitalwars.substack.com/p/impact-of-global-liquidity-on-bitcoin",
        "source_type": "blog",
        "citation": "Capital Wars (2025). Impact of Global Liquidity on Bitcoin.",
        "status": "new",
        "created_at": now_iso,
        "updated_at": now_iso,
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

# Append to JSONL backlog
with open("data/signal_research_backlog.jsonl", "a") as f:
    for c in candidates:
        f.write(json.dumps(c) + "\n")

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
db.execute(
    "CREATE INDEX IF NOT EXISTS idx_candidates_status ON signal_candidates(status)"
)
db.execute(
    "CREATE INDEX IF NOT EXISTS idx_candidates_score ON signal_candidates(composite_score DESC)"
)

for c in candidates:
    db.execute(
        """INSERT INTO signal_candidates
        (date,name,asset_class,target_assets,category,description,formula,
         data_sources,parameters,novelty_score,edge_evidence,data_availability,
         implementation_cost,non_redundancy,composite_score,source_url,source_type,
         citation,status,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            c["date"], c["name"], c["asset_class"],
            json.dumps(c["target_assets"]), c["category"], c["description"],
            c["formula"], json.dumps(c["data_sources"]),
            json.dumps(c["parameters"]), c["novelty_score"], c["edge_evidence"],
            c["data_availability"], c["implementation_cost"], c["non_redundancy"],
            c["composite_score"], c["source_url"], c["source_type"], c["citation"],
            c["status"], c["created_at"], c["updated_at"],
        ),
    )
db.commit()

# Print ranking
print("=== NEW CANDIDATE RANKING ===")
for c in sorted(candidates, key=lambda x: x["composite_score"], reverse=True):
    print(
        f"  {c['name']}: {c['composite_score']:.2f} "
        f"(N={c['novelty_score']}, E={c['edge_evidence']}, "
        f"D={c['data_availability']}, I={c['implementation_cost']}, "
        f"R={c['non_redundancy']})"
    )

# Check top from full backlog
print("\n=== TOP BACKLOG (status=new, score>=7.0) ===")
rows = db.execute(
    "SELECT name, composite_score FROM signal_candidates "
    "WHERE status='new' AND composite_score >= 7.0 "
    "ORDER BY composite_score DESC LIMIT 10"
).fetchall()
for r in rows:
    print(f"  {r[0]}: {r[1]:.2f}")

db.close()

# Write ranked output
ranked = {
    "date": date_str,
    "new_candidates_scored": len(candidates),
    "backlog_candidates_checked": len(rows),
    "top_candidate": {
        "name": "realized_skewness_directional",
        "composite_score": candidates[0]["composite_score"],
        "scores": {
            "novelty": 9.0, "edge_evidence": 8.0,
            "data_availability": 10.0, "implementation_cost": 9.0,
            "non_redundancy": 9.0,
        },
        "description": candidates[0]["description"],
        "formula": candidates[0]["formula"],
        "source": candidates[0]["citation"],
    },
    "all_ranked": [
        f"{c['name']}: {c['composite_score']:.2f}"
        for c in sorted(candidates, key=lambda x: x["composite_score"], reverse=True)
    ],
    "implementation_decision": "implement",
    "skip_reason": None,
}
pathlib.Path("data/signal_research_ranked.json").write_text(
    json.dumps(ranked, indent=2)
)

print("\nDone. All files written.")
