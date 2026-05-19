"""Historical backtest for Fix 3 (dxy_cross_asset) and Fix 1 (intraday
metals_cross_asset) signals.

Replays the signal logic against yfinance 60m historical bars to estimate
directional accuracy WITHOUT waiting weeks for live data.

Method:
1. Pull 60m OHLC bars for XAG (via SI=F futures proxy), DXY, copper, SPY, oil,
   and gold (for G/S ratio) for the last 60 days (the yfinance intraday cap).
2. For each bar t, compute each signal's vote using the SAME logic as
   production. Record the vote and the actual XAG outcome at t+1h, t+3h.
3. Aggregate: accuracy = correct_votes / total_directional_votes (HOLD
   excluded from denominator). Break down by horizon and signal.

Caveats:
- SI=F (silver futures) is a stand-in for spot XAG-USD during backtesting.
  Directional correlation is ~1.0 so it's adequate for accuracy estimation.
- yfinance 60m history is capped at ~730 days but only usable for the most
  recent 60 days without paid tier — we use 60 days.
- GVZ daily z-score is held constant at 0 for simplicity (GVZ has no intraday
  source, so the production signal also uses stale-daily). This removes the
  GVZ sub-signal from the backtest but it's 1 of 6 votes so not decisive.
- G/S ratio z-score is computed on a rolling 20-day window of daily closes
  to match production.

Output:
- Prints per-signal per-horizon accuracy to stdout.
- Writes data/backtest_new_signals_out.json with detailed metrics.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

# Make the repo importable so we can exercise the real signal code on
# historical inputs (rather than re-implementing thresholds and drifting).
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Signal imports (real production code).
from portfolio.signals.dxy_cross_asset import (  # noqa: E402
    _DXY_1H_THRESHOLD_PCT,
    _DXY_1H_CONFIDENCE_CAP_PCT,
)
from portfolio.signals.metals_cross_asset import (  # noqa: E402
    _COPPER_MOVE_INTRADAY_PCT,
    _SPY_MOVE_INTRADAY_PCT,
    _OIL_MOVE_INTRADAY_PCT,
    _GS_VELOCITY_INTRADAY_PCT,
    _GS_RATIO_ZSCORE,
)

OUTPUT = REPO / "data" / "backtest_new_signals_out.json"
BACKTEST_DAYS = 60  # yfinance intraday limit
BACKTEST_INTERVAL = "60m"

# Each horizon is an integer number of 60m bars forward.
HORIZONS = {"1h": 1, "3h": 3, "12h": 12, "1d": 24}

# Directional-outcome threshold: a move of at least THIS percent counts as
# a directional move. Below it, the outcome is treated as flat and BOTH
# BUY and SELL votes are scored as wrong (no credit for lucky flat windows).
# Matches the 0.1% threshold used by the existing XAG accuracy queries.
OUTCOME_THRESHOLD_PCT = 0.1


def _load_close(ticker: str, period_days: int, interval: str) -> pd.Series | None:
    """Download 60m bars and return the Close series aligned by timestamp."""
    period = f"{period_days}d"
    df = yf.download(
        ticker, period=period, interval=interval,
        progress=False, auto_adjust=True,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.empty or "Close" not in df.columns:
        print(f"  {ticker}: no data")
        return None
    s = df["Close"].dropna()
    # yfinance may return a tz-aware index in UTC — standardize.
    if s.index.tz is not None:
        s.index = s.index.tz_convert("UTC")
    s.name = ticker
    print(f"  {ticker}: {len(s)} bars, {s.index[0]} to {s.index[-1]}")
    return s


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Vectorized N-bar pct change as percentage."""
    return (series / series.shift(periods) - 1) * 100


def _dxy_vote(change_1h: float) -> tuple[str, float]:
    """Replay the dxy_cross_asset logic exactly as production does it."""
    if math.isnan(change_1h):
        return "HOLD", 0.0
    if change_1h < -_DXY_1H_THRESHOLD_PCT:
        return "BUY", min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
    if change_1h > _DXY_1H_THRESHOLD_PCT:
        return "SELL", min(abs(change_1h) / _DXY_1H_CONFIDENCE_CAP_PCT, 1.0)
    return "HOLD", 0.0


def _metals_cross_asset_vote(
    *, copper_3h: float, spy_3h: float, oil_3h: float, gs_vel_3h: float,
    gs_ratio_zscore: float, is_silver: bool = True,
) -> str:
    """Replay the intraday path of metals_cross_asset majority vote.

    GVZ is omitted (no intraday source — treated as HOLD).
    """
    votes: list[str] = []

    # Sub 1: copper
    if not math.isnan(copper_3h):
        if copper_3h > _COPPER_MOVE_INTRADAY_PCT:
            votes.append("BUY")
        elif copper_3h < -_COPPER_MOVE_INTRADAY_PCT:
            votes.append("SELL")
        else:
            votes.append("HOLD")
    else:
        votes.append("HOLD")

    # Sub 2: GVZ — held HOLD (not in intraday backtest)
    votes.append("HOLD")

    # Sub 3: G/S ratio z-score (silver-perspective)
    if is_silver:
        if gs_ratio_zscore > _GS_RATIO_ZSCORE:
            votes.append("BUY")
        elif gs_ratio_zscore < -_GS_RATIO_ZSCORE:
            votes.append("SELL")
        else:
            votes.append("HOLD")
    else:
        if gs_ratio_zscore > _GS_RATIO_ZSCORE:
            votes.append("HOLD")
        elif gs_ratio_zscore < -_GS_RATIO_ZSCORE:
            votes.append("BUY")
        else:
            votes.append("HOLD")

    # Sub 4: G/S velocity (silver-perspective)
    if not math.isnan(gs_vel_3h):
        if is_silver:
            if gs_vel_3h < -_GS_VELOCITY_INTRADAY_PCT:
                votes.append("BUY")
            elif gs_vel_3h > _GS_VELOCITY_INTRADAY_PCT:
                votes.append("SELL")
            else:
                votes.append("HOLD")
        else:
            if gs_vel_3h > _GS_VELOCITY_INTRADAY_PCT:
                votes.append("BUY")
            elif gs_vel_3h < -_GS_VELOCITY_INTRADAY_PCT:
                votes.append("SELL")
            else:
                votes.append("HOLD")
    else:
        votes.append("HOLD")

    # Sub 5: SPY risk-on/off
    if not math.isnan(spy_3h):
        if spy_3h > _SPY_MOVE_INTRADAY_PCT:
            votes.append("BUY" if is_silver else "HOLD")
        elif spy_3h < -_SPY_MOVE_INTRADAY_PCT:
            votes.append("BUY" if not is_silver else "SELL")
        else:
            votes.append("HOLD")
    else:
        votes.append("HOLD")

    # Sub 6: oil
    if not math.isnan(oil_3h):
        if oil_3h > _OIL_MOVE_INTRADAY_PCT:
            votes.append("BUY")
        elif oil_3h < -_OIL_MOVE_INTRADAY_PCT:
            votes.append("SELL")
        else:
            votes.append("HOLD")
    else:
        votes.append("HOLD")

    buys = votes.count("BUY")
    sells = votes.count("SELL")
    if buys > sells:
        return "BUY"
    if sells > buys:
        return "SELL"
    return "HOLD"


def _score_vote(vote: str, outcome_pct: float, threshold: float = OUTCOME_THRESHOLD_PCT) -> int | None:
    """Return 1 if correct, 0 if wrong, None if vote is HOLD or outcome is NaN."""
    if vote == "HOLD" or math.isnan(outcome_pct):
        return None
    if vote == "BUY":
        return 1 if outcome_pct > threshold else 0
    # SELL
    return 1 if outcome_pct < -threshold else 0


def main() -> None:
    print(f"Loading {BACKTEST_DAYS} days of {BACKTEST_INTERVAL} bars...")
    xag = _load_close("SI=F", BACKTEST_DAYS, BACKTEST_INTERVAL)
    dxy = _load_close("DX-Y.NYB", BACKTEST_DAYS, BACKTEST_INTERVAL)
    copper = _load_close("HG=F", BACKTEST_DAYS, BACKTEST_INTERVAL)
    spy = _load_close("SPY", BACKTEST_DAYS, BACKTEST_INTERVAL)
    oil = _load_close("CL=F", BACKTEST_DAYS, BACKTEST_INTERVAL)
    gold = _load_close("GC=F", BACKTEST_DAYS, BACKTEST_INTERVAL)

    missing = [n for n, s in [
        ("SI=F", xag), ("DX-Y.NYB", dxy), ("HG=F", copper),
        ("SPY", spy), ("CL=F", oil), ("GC=F", gold),
    ] if s is None]
    if missing:
        print(f"ERROR: missing data for {missing}; aborting")
        sys.exit(1)

    # Align to XAG's timestamps; forward-fill the others (SPY trades only
    # 9:30-16:00 ET so its timestamps are a strict subset of commodity
    # futures). Inner-join would leave 0 rows because SPY hours ≠ futures
    # hours. Forward-fill is the right behavior: during non-US hours, the
    # last known SPY close is what any signal would observe anyway.
    raw = pd.concat(
        {"xag": xag, "dxy": dxy, "copper": copper, "spy": spy, "oil": oil, "gold": gold},
        axis=1, join="outer",
    ).sort_index()
    # Keep only XAG-present rows; forward-fill the rest from their most
    # recent observation.
    df = raw.ffill().loc[xag.index].dropna()
    print(f"\nAligned bars: {len(df)} (spans {df.index[0]} to {df.index[-1]})")

    # --- Compute per-bar inputs ---
    df["dxy_change_1h"] = _pct_change(df["dxy"], 1)
    df["copper_change_3h"] = _pct_change(df["copper"], 3)
    df["spy_change_3h"] = _pct_change(df["spy"], 3)
    df["oil_change_3h"] = _pct_change(df["oil"], 3)

    # Gold/silver ratio intraday velocity
    df["gs_ratio"] = df["gold"] / df["xag"]
    df["gs_velocity_3h"] = _pct_change(df["gs_ratio"], 3)

    # Daily G/S ratio z-score (20 daily closes). Approximate from 60m bars
    # by resampling to 1D and computing rolling z-score.
    gs_daily = df["gs_ratio"].resample("1D").last().dropna()
    gs_daily_mean20 = gs_daily.rolling(20, min_periods=10).mean()
    gs_daily_std20 = gs_daily.rolling(20, min_periods=10).std()
    gs_daily_zscore = (gs_daily - gs_daily_mean20) / gs_daily_std20
    # Forward-fill daily z-score into each 60m bar.
    df["gs_ratio_zscore"] = gs_daily_zscore.reindex(df.index, method="ffill").fillna(0.0)

    # --- XAG outcomes for each horizon ---
    outcomes: dict[str, pd.Series] = {}
    for h_name, bars in HORIZONS.items():
        outcomes[h_name] = (df["xag"].shift(-bars) / df["xag"] - 1) * 100

    # --- Iterate bars ---
    stats = {
        sig: {h: {"correct": 0, "total_dir": 0, "buy": 0, "sell": 0, "hold": 0}
              for h in HORIZONS}
        for sig in ("dxy_cross_asset", "metals_cross_asset")
    }

    for ts, row in df.iterrows():
        dxy_vote, _dxy_conf = _dxy_vote(row["dxy_change_1h"])
        mca_vote = _metals_cross_asset_vote(
            copper_3h=row["copper_change_3h"],
            spy_3h=row["spy_change_3h"],
            oil_3h=row["oil_change_3h"],
            gs_vel_3h=row["gs_velocity_3h"],
            gs_ratio_zscore=row["gs_ratio_zscore"],
            is_silver=True,
        )

        for h_name in HORIZONS:
            out = outcomes[h_name].loc[ts] if ts in outcomes[h_name].index else float("nan")

            for sig_name, vote in [
                ("dxy_cross_asset", dxy_vote),
                ("metals_cross_asset", mca_vote),
            ]:
                s = stats[sig_name][h_name]
                s[vote.lower()] += 1
                score = _score_vote(vote, out)
                if score is None:
                    continue
                s["total_dir"] += 1
                if score == 1:
                    s["correct"] += 1

    # --- Report ---
    out_doc: dict[str, Any] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "backtest_days": BACKTEST_DAYS,
        "bars_used": len(df),
        "outcome_threshold_pct": OUTCOME_THRESHOLD_PCT,
        "results": {},
    }

    print()
    print("=" * 68)
    print(f"{'signal':<22} {'horizon':>7} {'acc':>7} {'n_dir':>6} {'buy':>5} {'sell':>5} {'hold':>5}")
    print("-" * 68)
    for sig in ("dxy_cross_asset", "metals_cross_asset"):
        out_doc["results"][sig] = {}
        for h_name in HORIZONS:
            s = stats[sig][h_name]
            acc = (s["correct"] / s["total_dir"] * 100) if s["total_dir"] else float("nan")
            out_doc["results"][sig][h_name] = {
                "accuracy_pct": None if math.isnan(acc) else round(acc, 2),
                "correct": s["correct"],
                "total_directional": s["total_dir"],
                "buy_count": s["buy"],
                "sell_count": s["sell"],
                "hold_count": s["hold"],
            }
            marker = (
                "++" if not math.isnan(acc) and acc >= 55 else
                ("--" if not math.isnan(acc) and acc < 45 else "")
            )
            acc_str = f"{acc:>6.1f}%" if not math.isnan(acc) else "   N/A"
            print(f"{sig:<22} {h_name:>7} {acc_str} {s['total_dir']:>6} "
                  f"{s['buy']:>5} {s['sell']:>5} {s['hold']:>5}  {marker}")

    print("=" * 68)

    # Save
    OUTPUT.write_text(json.dumps(out_doc, indent=2))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
