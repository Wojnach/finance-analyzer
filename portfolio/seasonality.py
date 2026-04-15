"""Intraday seasonality detrending for metals and crypto.

Computes average return and volatility profiles by hour-of-day from
historical data, then subtracts these patterns from current observations
to isolate non-seasonal signal content.

Research basis: Smales & Yang (2015) — removing day-cycle (detrending)
sharpens short-term signals for gold and silver.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.seasonality")

_BASE_DIR = Path(__file__).resolve().parent.parent
_STATE_FILE = _BASE_DIR / "data" / "seasonality_profiles.json"
_MIN_DAYS = 5  # minimum trading days to compute profiles


def compute_hourly_profile(klines_1h: pd.DataFrame) -> dict | None:
    """Compute average return and volatility by hour-of-day.

    Args:
        klines_1h: DataFrame with 'close' column and DatetimeIndex (1h bars).
                   Needs at least _MIN_DAYS * 24 rows.

    Returns:
        Dict keyed by hour (0-23), each with mean_return and mean_volatility.
        None if insufficient data.
    """
    if klines_1h is None or len(klines_1h) < _MIN_DAYS * 24:
        return None

    df = klines_1h.copy()
    df["return"] = df["close"].pct_change()
    df["abs_return"] = df["return"].abs()

    # Extract hour from index
    if hasattr(df.index, "hour"):
        df["hour"] = df.index.hour
    else:
        return None

    # Group by hour and compute mean return + mean absolute return (vol proxy)
    grouped = df.groupby("hour").agg(
        mean_return=("return", "mean"),
        mean_volatility=("abs_return", "mean"),
        count=("return", "count"),
    )

    profile = {}
    for hour in range(24):
        if hour in grouped.index:
            row = grouped.loc[hour]
            profile[str(hour)] = {
                "mean_return": float(row["mean_return"]),
                "mean_volatility": float(row["mean_volatility"]),
                "count": int(row["count"]),
            }
        else:
            profile[str(hour)] = {
                "mean_return": 0.0,
                "mean_volatility": 0.0,
                "count": 0,
            }

    return profile


def detrend_return(raw_return: float, hour: int, profile: dict) -> float:
    """Remove seasonal component from a return observation.

    Args:
        raw_return: The observed return (e.g. 0.002 for 0.2%).
        hour: Hour of day (0-23, UTC).
        profile: Hourly profile dict from compute_hourly_profile.

    Returns:
        Detrended return: raw_return - mean_return_for_hour.
    """
    if profile is None:
        return raw_return
    entry = profile.get(str(hour))
    if entry is None:
        return raw_return
    return raw_return - entry["mean_return"]


def normalize_volatility(raw_vol: float, hour: int, profile: dict) -> float:
    """Normalize volatility by dividing by the seasonal average for this hour.

    Args:
        raw_vol: Observed absolute return or volatility measure.
        hour: Hour of day (0-23, UTC).
        profile: Hourly profile dict from compute_hourly_profile.

    Returns:
        Normalized volatility (1.0 = average for this hour).
        Returns raw_vol if profile unavailable.
    """
    if profile is None:
        return raw_vol
    entry = profile.get(str(hour))
    if entry is None or entry["mean_volatility"] < 1e-10:
        return raw_vol
    return raw_vol / entry["mean_volatility"]


def save_profiles(profiles: dict[str, dict]) -> None:
    """Persist ticker-keyed profiles to disk.

    Args:
        profiles: Dict keyed by ticker, each value is an hourly profile.
    """
    atomic_write_json(_STATE_FILE, profiles)


def load_profiles() -> dict:
    """Load persisted profiles from disk."""
    return load_json(_STATE_FILE) or {}


def get_profile(ticker: str) -> dict | None:
    """Load the hourly profile for a specific ticker."""
    profiles = load_profiles()
    return profiles.get(ticker)
