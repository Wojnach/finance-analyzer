"""Grid-search threshold tuning for dxy_cross_asset + metals_cross_asset.

Runs the same historical backtest as scripts/backtest_new_signals.py, but
evaluates multiple threshold sets to find the one with the best risk-
adjusted accuracy.

Scoring metric (per horizon): accuracy × log(1 + n_directional). The log
discourages configurations that fire once and happen to be right — we
need enough fires to be confident.

Output: prints a sorted table, writes data/tune_new_signals_out.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUTPUT = REPO / "data" / "tune_new_signals_out.json"
BACKTEST_DAYS = 60
INTERVAL = "60m"
HORIZONS = {"1h": 1, "3h": 3, "12h": 12, "1d": 24}
OUTCOME_THRESHOLD_PCT = 0.1


def _load_close(ticker: str) -> pd.Series:
    df = yf.download(ticker, period=f"{BACKTEST_DAYS}d", interval=INTERVAL,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    s = df["Close"].dropna()
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC")
    return s


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return (series / series.shift(periods) - 1) * 100


def _score_vote(vote: str, out: float) -> int | None:
    if vote == "HOLD" or math.isnan(out):
        return None
    if vote == "BUY":
        return 1 if out > OUTCOME_THRESHOLD_PCT else 0
    return 1 if out < -OUTCOME_THRESHOLD_PCT else 0


def _dxy_votes(series: pd.Series, thr: float) -> pd.Series:
    """Vectorized dxy_cross_asset vote given a 1h change series."""
    votes = pd.Series("HOLD", index=series.index)
    votes[series < -thr] = "BUY"
    votes[series > thr] = "SELL"
    return votes


def _metals_votes(
    df: pd.DataFrame, *, copper_thr: float, spy_thr: float, oil_thr: float,
    gs_vel_thr: float, gs_z_thr: float = 1.5,
) -> pd.Series:
    """Vectorized metals_cross_asset vote (silver perspective only).

    GVZ omitted (no intraday); so effectively a 5-sub-signal majority vote.
    """
    # Sub-signal votes per bar
    copper_vote = pd.Series("HOLD", index=df.index)
    copper_vote[df["copper_change_3h"] > copper_thr] = "BUY"
    copper_vote[df["copper_change_3h"] < -copper_thr] = "SELL"

    gs_z_vote = pd.Series("HOLD", index=df.index)
    gs_z_vote[df["gs_ratio_zscore"] > gs_z_thr] = "BUY"
    gs_z_vote[df["gs_ratio_zscore"] < -gs_z_thr] = "SELL"

    gs_vel_vote = pd.Series("HOLD", index=df.index)
    gs_vel_vote[df["gs_velocity_3h"] < -gs_vel_thr] = "BUY"
    gs_vel_vote[df["gs_velocity_3h"] > gs_vel_thr] = "SELL"

    spy_vote = pd.Series("HOLD", index=df.index)
    spy_vote[df["spy_change_3h"] > spy_thr] = "BUY"
    spy_vote[df["spy_change_3h"] < -spy_thr] = "SELL"

    oil_vote = pd.Series("HOLD", index=df.index)
    oil_vote[df["oil_change_3h"] > oil_thr] = "BUY"
    oil_vote[df["oil_change_3h"] < -oil_thr] = "SELL"

    # Tally per bar
    sub_df = pd.DataFrame({
        "copper": copper_vote, "gs_z": gs_z_vote, "gs_vel": gs_vel_vote,
        "spy": spy_vote, "oil": oil_vote,
    })
    buys = (sub_df == "BUY").sum(axis=1)
    sells = (sub_df == "SELL").sum(axis=1)
    result = pd.Series("HOLD", index=df.index)
    result[buys > sells] = "BUY"
    result[sells > buys] = "SELL"
    return result


def _evaluate(votes: pd.Series, outcomes: dict[str, pd.Series]) -> dict:
    """Compute per-horizon accuracy metrics for a vote series."""
    out_metrics = {}
    for h, out_series in outcomes.items():
        correct = total_dir = 0
        for ts, vote in votes.items():
            if ts not in out_series.index:
                continue
            sc = _score_vote(vote, out_series.loc[ts])
            if sc is None:
                continue
            total_dir += 1
            correct += sc
        acc = correct / total_dir if total_dir else 0
        # Score: accuracy weighted by log-coverage. Rewards being right often.
        score = acc * math.log(1 + total_dir) if total_dir else 0
        out_metrics[h] = {
            "accuracy_pct": round(acc * 100, 2) if total_dir else None,
            "n_directional": total_dir,
            "correct": correct,
            "score": round(score, 4),
        }
    return out_metrics


def main() -> None:
    print(f"Loading {BACKTEST_DAYS}d of {INTERVAL} bars...")
    xag = _load_close("SI=F")
    dxy = _load_close("DX-Y.NYB")
    copper = _load_close("HG=F")
    spy = _load_close("SPY")
    oil = _load_close("CL=F")
    gold = _load_close("GC=F")

    raw = pd.concat(
        {"xag": xag, "dxy": dxy, "copper": copper, "spy": spy, "oil": oil, "gold": gold},
        axis=1, join="outer",
    ).sort_index()
    df = raw.ffill().loc[xag.index].dropna()
    print(f"Aligned {len(df)} bars\n")

    df["dxy_change_1h"] = _pct_change(df["dxy"], 1)
    df["copper_change_3h"] = _pct_change(df["copper"], 3)
    df["spy_change_3h"] = _pct_change(df["spy"], 3)
    df["oil_change_3h"] = _pct_change(df["oil"], 3)
    df["gs_ratio"] = df["gold"] / df["xag"]
    df["gs_velocity_3h"] = _pct_change(df["gs_ratio"], 3)

    gs_daily = df["gs_ratio"].resample("1D").last().dropna()
    gs_zscore_daily = (gs_daily - gs_daily.rolling(20, min_periods=10).mean()) / gs_daily.rolling(20, min_periods=10).std()
    df["gs_ratio_zscore"] = gs_zscore_daily.reindex(df.index, method="ffill").fillna(0.0)

    outcomes = {h: (df["xag"].shift(-b) / df["xag"] - 1) * 100 for h, b in HORIZONS.items()}

    results: list[dict] = []

    # --- DXY threshold sweep ---
    print("=" * 72)
    print("DXY cross-asset threshold sweep")
    print("=" * 72)
    print(f"{'dxy_thr':>8} {'1h':>7} {'3h':>7} {'12h':>7} {'1d':>7} {'n_dir':>7}")
    for thr in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]:
        votes = _dxy_votes(df["dxy_change_1h"], thr)
        metrics = _evaluate(votes, outcomes)
        n_dir = metrics["1h"]["n_directional"]
        row = f"{thr:>7.2f}% "
        for h in ("1h", "3h", "12h", "1d"):
            acc = metrics[h]["accuracy_pct"]
            row += f"{acc:>6.1f}% " if acc is not None else "   N/A "
        row += f"{n_dir:>6}"
        print(row)
        results.append({
            "signal": "dxy_cross_asset",
            "thresholds": {"dxy_1h_thr": thr},
            "metrics": metrics,
        })

    # --- Metals cross-asset threshold sweep ---
    # Tuple: (copper_thr, spy_thr, oil_thr, gs_vel_thr)
    print()
    print("=" * 72)
    print("metals_cross_asset threshold sweep (current: 0.4, 0.25, 0.5, 0.5)")
    print("=" * 72)
    print(f"{'cu':>5} {'spy':>5} {'oil':>5} {'gsvel':>5} "
          f"{'1h':>7} {'3h':>7} {'12h':>7} {'1d':>7} {'n3h':>5}")

    # Grid: vary each threshold. Keep grid small to avoid runtime explosion.
    grids = [
        # Baseline (current production)
        (0.4, 0.25, 0.5, 0.5),
        # Tighter across the board (fewer fires, higher per-fire accuracy hopefully)
        (0.6, 0.4, 0.8, 0.7),
        (0.8, 0.5, 1.0, 1.0),
        (1.0, 0.7, 1.2, 1.2),
        # Looser (more fires)
        (0.3, 0.2, 0.4, 0.4),
        (0.2, 0.15, 0.3, 0.3),
        # Asymmetric — tight on noisy sources, loose on clean
        (0.6, 0.3, 0.6, 0.5),
        (0.5, 0.4, 0.6, 0.6),
        (0.7, 0.3, 0.7, 0.5),
    ]
    for (cu, sp, oi, gv) in grids:
        votes = _metals_votes(
            df, copper_thr=cu, spy_thr=sp, oil_thr=oi, gs_vel_thr=gv,
        )
        metrics = _evaluate(votes, outcomes)
        n3h = metrics["3h"]["n_directional"]
        row = f"{cu:>5.2f} {sp:>5.2f} {oi:>5.2f} {gv:>5.2f}"
        for h in ("1h", "3h", "12h", "1d"):
            acc = metrics[h]["accuracy_pct"]
            row += f" {acc:>6.1f}%" if acc is not None else "    N/A"
        row += f" {n3h:>5}"
        print(row)
        results.append({
            "signal": "metals_cross_asset",
            "thresholds": {
                "copper": cu, "spy": sp, "oil": oi, "gs_velocity": gv,
            },
            "metrics": metrics,
        })

    OUTPUT.write_text(json.dumps({
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "backtest_days": BACKTEST_DAYS,
        "bars_used": len(df),
        "results": results,
    }, indent=2))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
