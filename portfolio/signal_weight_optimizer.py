"""Walk-forward signal weight optimizer.

Retrains signal weights using rolling windows to prevent overfitting
and adapt to changing market regimes. Uses the LinearFactorModel for
per-window ridge regression.

Walk-forward method:
    1. Split history into train/test windows (e.g. 30d train, 7d test)
    2. Train model on each window, score on out-of-sample test period
    3. Track per-signal weight stability and out-of-sample performance
    4. Output: recommended weights and stability metrics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.linear_factor import LinearFactorModel

logger = logging.getLogger("portfolio.signal_weight_optimizer")

_RESULTS_FILE = Path("data/models/walkforward_results.json")


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward optimization run."""
    n_windows: int = 0
    avg_r_squared: float = 0.0
    avg_oos_corr: float = 0.0  # out-of-sample correlation
    weight_stability: dict[str, float] = field(default_factory=dict)
    recommended_weights: dict[str, float] = field(default_factory=dict)
    signal_rankings: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_windows": self.n_windows,
            "avg_r_squared": self.avg_r_squared,
            "avg_oos_corr": self.avg_oos_corr,
            "weight_stability": self.weight_stability,
            "recommended_weights": self.recommended_weights,
            "signal_rankings": self.signal_rankings,
        }


def walk_forward_optimize(
    signals_df: pd.DataFrame,
    returns: pd.Series,
    train_window: int = 720,   # 720 hours = 30 days
    test_window: int = 168,    # 168 hours = 7 days
    step_size: int = 168,      # step by 7 days
    alpha: float = 1.0,
    min_train_samples: int = 100,
) -> WalkForwardResult:
    """Run walk-forward optimization across rolling windows.

    Args:
        signals_df: DataFrame of signal values (columns=signals, rows=time).
        returns: Series of forward returns aligned with signals_df.
        train_window: Number of rows for training period.
        test_window: Number of rows for test period.
        step_size: Step size between windows.
        alpha: Ridge regularization strength.
        min_train_samples: Minimum training samples per window.

    Returns:
        WalkForwardResult with averaged metrics and recommended weights.
    """
    common = signals_df.index.intersection(returns.index)
    signals_df = signals_df.loc[common]
    returns = returns.loc[common]
    n = len(common)

    if n < train_window + test_window:
        logger.warning("Insufficient data for walk-forward: %d < %d",
                      n, train_window + test_window)
        return WalkForwardResult()

    all_weights: list[dict[str, float]] = []
    r_squared_scores: list[float] = []
    oos_correlations: list[float] = []

    start = 0
    while start + train_window + test_window <= n:
        train_end = start + train_window
        test_end = train_end + test_window

        train_X = signals_df.iloc[start:train_end]
        train_y = returns.iloc[start:train_end]
        test_X = signals_df.iloc[train_end:test_end]
        test_y = returns.iloc[train_end:test_end]

        model = LinearFactorModel(alpha=alpha)
        if not model.fit(train_X, train_y, min_samples=min_train_samples):
            start += step_size
            continue

        r_squared_scores.append(model.r_squared)
        all_weights.append(model.weights)

        # Out-of-sample prediction correlation
        predictions = []
        for _, row in test_X.iterrows():
            predictions.append(model.predict(row.to_dict()))
        if len(predictions) > 1 and test_y.std() > 1e-10:
            corr = float(np.corrcoef(predictions, test_y.values)[0, 1])
            if not np.isnan(corr):
                oos_correlations.append(corr)

        start += step_size

    if not all_weights:
        return WalkForwardResult()

    # Compute weight stability: std of each weight across windows / mean of abs
    all_signals = set()
    for w in all_weights:
        all_signals.update(w.keys())

    weight_stability = {}
    recommended_weights = {}
    for sig in all_signals:
        values = [w.get(sig, 0.0) for w in all_weights]
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        mean_abs = float(np.mean(np.abs(values)))
        # Stability = 1 - (std / mean_abs). High = consistent direction.
        stability = 1.0 - (std_val / mean_abs) if mean_abs > 1e-10 else 0.0
        weight_stability[sig] = round(max(0.0, stability), 4)
        recommended_weights[sig] = round(mean_val, 6)

    # Rank signals by |mean_weight| * stability
    signal_rankings = sorted(
        [(sig, round(abs(recommended_weights[sig]) * weight_stability.get(sig, 0), 6))
         for sig in all_signals],
        key=lambda x: x[1],
        reverse=True,
    )

    result = WalkForwardResult(
        n_windows=len(all_weights),
        avg_r_squared=round(float(np.mean(r_squared_scores)), 4),
        avg_oos_corr=round(float(np.mean(oos_correlations)), 4) if oos_correlations else 0.0,
        weight_stability=weight_stability,
        recommended_weights=recommended_weights,
        signal_rankings=signal_rankings,
    )
    return result


def save_results(result: WalkForwardResult, path: Path | None = None) -> None:
    """Persist walk-forward results to JSON."""
    path = path or _RESULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, result.to_dict())


def load_results(path: Path | None = None) -> WalkForwardResult | None:
    """Load walk-forward results from JSON."""
    path = path or _RESULTS_FILE
    data = load_json(path)
    if not data:
        return None
    return WalkForwardResult(**data)
