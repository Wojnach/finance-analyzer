"""Signal correlation & HOLD dilution audit.

Reads the last 500 signal_log.jsonl entries, computes pairwise agreement
rates between signals, identifies redundancy clusters, and quantifies
HOLD dilution from disabled/permanently-inactive signals.

Output: data/daily_research_signal_audit.json
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TARGET_TICKERS = ["BTC-USD", "ETH-USD", "XAG-USD", "XAU-USD"]

# Hard-coded disabled set (from tickers.py)
DISABLED_SIGNALS = {
    "ml", "futures_basis", "hurst_regime", "shannon_entropy",
    "vix_term_structure", "gold_real_yield_paradox", "cross_asset_tsmom",
    "copper_gold_ratio", "statistical_jump_regime", "network_momentum",
    "ovx_metals_spillover", "xtrend_equity_spillover", "complexity_gap_regime",
    "realized_skewness", "mahalanobis_turbulence", "crypto_evrp",
    "orderbook_flow", "smart_money",
}


def load_entries(n=500):
    with open(DATA_DIR / "signal_log.jsonl") as f:
        all_lines = f.readlines()
    lines = all_lines[-n:]
    entries = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


def analyse_hold_dilution(ticker_votes, signal_names_per_ticker):
    hold_dilution = {}
    for tk in TARGET_TICKERS:
        if tk not in ticker_votes:
            continue
        snapshots = ticker_votes[tk]
        total_snapshots = len(snapshots)

        signal_hold_rate = {}
        for sig in signal_names_per_ticker[tk]:
            hold_count = sum(1 for snap in snapshots if snap.get(sig) == "HOLD")
            signal_hold_rate[sig] = hold_count / total_snapshots if total_snapshots else 0

        permanent_holds = {s: r for s, r in signal_hold_rate.items() if r > 0.95}
        active_signals = {s: r for s, r in signal_hold_rate.items() if r <= 0.95}

        avg_hold_pcts = []
        for snap in snapshots:
            total = len(snap)
            holds = sum(1 for v in snap.values() if v == "HOLD")
            avg_hold_pcts.append(holds / total * 100 if total else 0)

        avg_dilution = sum(avg_hold_pcts) / len(avg_hold_pcts) if avg_hold_pcts else 0

        # Categorize permanent HOLDs
        disabled_hold = sorted(s for s in permanent_holds if s in DISABLED_SIGNALS)
        accuracy_gated_hold = sorted(s for s in permanent_holds if s not in DISABLED_SIGNALS)

        hold_dilution[tk] = {
            "total_signals": len(signal_names_per_ticker[tk]),
            "permanent_hold_count": len(permanent_holds),
            "disabled_hold_signals": disabled_hold,
            "accuracy_gated_hold_signals": accuracy_gated_hold,
            "active_voter_count": len(active_signals),
            "active_voter_names": sorted(active_signals.keys()),
            "avg_hold_pct": round(avg_dilution, 1),
        }
    return hold_dilution


def analyse_correlations(ticker_votes, signal_names_per_ticker):
    correlation_clusters = {}

    for tk in TARGET_TICKERS:
        if tk not in ticker_votes:
            continue
        snapshots = ticker_votes[tk]

        # Only analyse signals active >5% of the time
        active_sigs = []
        for sig in signal_names_per_ticker[tk]:
            non_hold = sum(1 for snap in snapshots if snap.get(sig) in ("BUY", "SELL"))
            if non_hold >= len(snapshots) * 0.05:
                active_sigs.append(sig)
        active_sigs.sort()

        if len(active_sigs) < 2:
            correlation_clusters[tk] = {
                "active_signals": active_sigs,
                "high_agreement_pairs": [],
                "high_directional_pairs": [],
                "clusters": [],
            }
            continue

        # Pairwise agreement (all votes including HOLD)
        agreement = {}
        for s1, s2 in combinations(active_sigs, 2):
            agree = total = 0
            for snap in snapshots:
                v1, v2 = snap.get(s1), snap.get(s2)
                if v1 is not None and v2 is not None:
                    total += 1
                    if v1 == v2:
                        agree += 1
            if total:
                agreement[(s1, s2)] = agree / total

        # Directional agreement (excluding mutual HOLD)
        directional = {}
        for s1, s2 in combinations(active_sigs, 2):
            agree = total = 0
            for snap in snapshots:
                v1, v2 = snap.get(s1), snap.get(s2)
                if v1 is not None and v2 is not None:
                    if v1 != "HOLD" or v2 != "HOLD":
                        total += 1
                        if v1 == v2:
                            agree += 1
            if total:
                directional[(s1, s2)] = agree / total

        high_agreement = {k: v for k, v in agreement.items() if v > 0.80}

        # Greedy clustering of >80% agreement pairs
        clusters = []
        for (s1, s2), _ in sorted(high_agreement.items(), key=lambda x: -x[1]):
            found = None
            for c in clusters:
                if s1 in c or s2 in c:
                    c.add(s1)
                    c.add(s2)
                    found = c
                    break
            if found is None:
                clusters.append({s1, s2})

        # Merge overlapping
        merged = True
        while merged:
            merged = False
            new_clusters = []
            for c in clusters:
                placed = False
                for nc in new_clusters:
                    if nc & c:
                        nc |= c
                        placed = True
                        merged = True
                        break
                if not placed:
                    new_clusters.append(c)
            clusters = new_clusters

        correlation_clusters[tk] = {
            "active_signals": active_sigs,
            "high_agreement_pairs": [
                {"signal_1": s1, "signal_2": s2, "agreement_pct": round(r * 100, 1)}
                for (s1, s2), r in sorted(high_agreement.items(), key=lambda x: -x[1])
            ],
            "high_directional_pairs": [
                {"signal_1": s1, "signal_2": s2, "agreement_pct": round(r * 100, 1)}
                for (s1, s2), r in sorted(directional.items(), key=lambda x: -x[1])[:25]
            ],
            "clusters": [sorted(c) for c in clusters],
        }
    return correlation_clusters


def build_accuracy_rankings(acc_cache):
    acc_1d = acc_cache.get("1d", {})
    acc_3h = acc_cache.get("3h", {})
    acc_1d_recent = acc_cache.get("1d_recent", {})

    top_signals, worst_signals = [], []

    for sig, data in sorted(acc_1d.items(), key=lambda x: -x[1].get("pct", 0)):
        if data.get("total", 0) < 30:
            continue
        entry = {
            "signal": sig,
            "accuracy_1d_pct": data.get("pct", 0),
            "samples_1d": data.get("total", 0),
            "accuracy_3h_pct": acc_3h.get(sig, {}).get("pct", 0),
            "samples_3h": acc_3h.get(sig, {}).get("total", 0),
            "disabled": sig in DISABLED_SIGNALS,
        }
        if acc_1d_recent and sig in acc_1d_recent:
            entry["accuracy_1d_recent_pct"] = acc_1d_recent[sig].get("pct", 0)

        if data["pct"] >= 55:
            top_signals.append(entry)
        elif data["pct"] < 47:
            worst_signals.append(entry)

    return top_signals, worst_signals


def build_recommendations(hold_dilution, correlation_clusters, worst_signals, acc_1d):
    recs = []

    # HOLD dilution
    for tk, dil in hold_dilution.items():
        if dil["avg_hold_pct"] > 70:
            recs.append(
                f"CRITICAL: {tk} has {dil['avg_hold_pct']:.0f}% HOLD dilution - "
                f"{dil['permanent_hold_count']} of {dil['total_signals']} signals are permanent HOLD. "
                f"Only {dil['active_voter_count']} signals actively vote. "
                f"Exclude permanently-disabled signals from voter count."
            )

    # Redundant clusters
    for tk, cc in correlation_clusters.items():
        for cluster in cc.get("clusters", []):
            if len(cluster) >= 3:
                recs.append(
                    f"REDUNDANCY: {tk} cluster {cluster} - {len(cluster)} signals agree >80%. "
                    f"Consider composite or weight-sharing to reduce redundant votes."
                )
            elif len(cluster) == 2:
                recs.append(
                    f"REDUNDANCY: {tk} pair {cluster} agree >80%. Low marginal info from both."
                )

    # Below-gate signals still active
    for s in worst_signals:
        if not s["disabled"]:
            recs.append(
                f"ACCURACY: {s['signal']} at {s['accuracy_1d_pct']:.1f}% (1d, {s['samples_1d']} sam) - "
                f"below accuracy gate, should be force-HOLD or disabled."
            )

    # Zero-sample dead weight
    zero_sample = sorted(sig for sig, data in acc_1d.items() if data.get("total", 0) == 0)
    if zero_sample:
        recs.append(
            f"DEAD WEIGHT: {len(zero_sample)} signals have 0 accuracy samples and are permanent HOLD: "
            f"{zero_sample}. Remove from voter denominator."
        )

    return recs


def main():
    entries = load_entries(500)
    print(f"Loaded {len(entries)} signal log entries")

    # Collect per-ticker signal votes
    ticker_votes = defaultdict(list)
    signal_names_per_ticker = defaultdict(set)

    for entry in entries:
        tickers = entry.get("tickers", {})
        for tk in TARGET_TICKERS:
            if tk not in tickers:
                continue
            signals = tickers[tk].get("signals", {})
            if signals:
                ticker_votes[tk].append(signals)
                signal_names_per_ticker[tk].update(signals.keys())

    for tk in TARGET_TICKERS:
        if tk in ticker_votes:
            print(f"  {tk}: {len(ticker_votes[tk])} snapshots, {len(signal_names_per_ticker[tk])} signals")

    # Run analyses
    hold_dilution = analyse_hold_dilution(ticker_votes, signal_names_per_ticker)
    correlation_clusters = analyse_correlations(ticker_votes, signal_names_per_ticker)

    with open(DATA_DIR / "accuracy_cache.json") as f:
        acc_cache = json.load(f)

    top_signals, worst_signals = build_accuracy_rankings(acc_cache)
    recommendations = build_recommendations(
        hold_dilution, correlation_clusters, worst_signals, acc_cache.get("1d", {})
    )

    # Print summary
    print("\n=== HOLD DILUTION ===")
    for tk, dil in hold_dilution.items():
        print(f"  {tk}: {dil['avg_hold_pct']:.1f}% HOLD, "
              f"{dil['permanent_hold_count']}/{dil['total_signals']} permanent, "
              f"{dil['active_voter_count']} active")

    print("\n=== CORRELATION CLUSTERS ===")
    for tk, cc in correlation_clusters.items():
        print(f"  {tk}: {len(cc['active_signals'])} active, {len(cc['clusters'])} clusters")
        for pair in cc["high_agreement_pairs"][:5]:
            print(f"    {pair['signal_1']} <-> {pair['signal_2']}: {pair['agreement_pct']}%")
        for i, cluster in enumerate(cc["clusters"]):
            print(f"    Cluster {i+1}: {cluster}")

    print(f"\n=== TOP SIGNALS (>55% at 1d, 30+ samples) ===")
    for s in top_signals:
        print(f"  {s['signal']:25s}: {s['accuracy_1d_pct']:.1f}% (1d) / {s['accuracy_3h_pct']:.1f}% (3h)")

    print(f"\n=== WORST SIGNALS (<47% at 1d, 30+ samples) ===")
    for s in worst_signals:
        d = " [DISABLED]" if s["disabled"] else ""
        print(f"  {s['signal']:25s}: {s['accuracy_1d_pct']:.1f}% (1d) / {s['accuracy_3h_pct']:.1f}% (3h){d}")

    print(f"\n=== RECOMMENDATIONS ({len(recommendations)}) ===")
    for i, r in enumerate(recommendations, 1):
        print(f"  {i}. {r}")

    # Write output
    output = {
        "generated_at": entries[-1].get("ts", "") if entries else "",
        "analysis_window": f"last {len(entries)} signal_log entries",
        "top_signals": top_signals,
        "worst_signals": worst_signals,
        "correlation_clusters": correlation_clusters,
        "hold_dilution_stats": hold_dilution,
        "recommendations": recommendations,
    }

    outpath = DATA_DIR / "daily_research_signal_audit.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to {outpath}")


if __name__ == "__main__":
    main()
