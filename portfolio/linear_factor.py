"""Linear factor model for signal combination.

Alternative to majority voting: trains ridge regression weights on historical
signal snapshots and forward returns. Produces a continuous score rather than
BUY/SELL/HOLD vote counts.

Formula: predicted_return = B0 + sum_i Bi * zi
where zi are z-scored signal features.

Usage:
    model = LinearFactorModel()
    model.fit(signal_history_df, returns_series)
    score = model.predict(current_signals_dict)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.linear_factor")

_MODEL_FILE = Path("data/models/linear_factor_weights.json")


class LinearFactorModel:
    """Ridge regression model for combining trading signals.

    Attributes:
        weights: Dict mapping signal name to beta weight.
        intercept: beta_0 intercept term.
        alpha: Ridge regularization strength.
        feature_means: Dict of feature means for z-scoring.
        feature_stds: Dict of feature stds for z-scoring.
        r_squared: Training R-squared score.
        n_samples: Number of training samples used.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: dict[str, float] = {}
        self.intercept: float = 0.0
        self.feature_means: dict[str, float] = {}
        self.feature_stds: dict[str, float] = {}
        self.r_squared: float = 0.0
        self.n_samples: int = 0

    def fit(self, signals_df: pd.DataFrame, returns: pd.Series,
            min_samples: int = 30) -> bool:
        """Train ridge regression on historical signal data.

        Args:
            signals_df: DataFrame where columns are signal names and rows are
                       time observations. Values should be numeric
                       (e.g. confidence * direction_sign).
            returns: Series of forward returns aligned with signals_df index.
            min_samples: Minimum training samples required.

        Returns:
            True if training succeeded, False if insufficient data.
        """
        # Align and drop NaN
        common = signals_df.index.intersection(returns.index)
        if len(common) < min_samples:
            logger.warning("Insufficient data for linear factor: %d < %d",
                           len(common), min_samples)
            return False

        X = signals_df.loc[common].copy()
        y = returns.loc[common].copy()

        # Drop columns with zero variance
        stds = X.std()
        valid_cols = stds[stds > 1e-10].index.tolist()
        if not valid_cols:
            logger.warning("No valid signal columns (all zero variance)")
            return False
        X = X[valid_cols]

        # Z-score features
        means = X.mean()
        stds = X.std()
        X_z = (X - means) / stds.replace(0, 1)

        # Ridge regression: beta = (X'X + alpha*I)^-1 X'y
        X_arr = X_z.values
        y_arr = y.values
        n_features = X_arr.shape[1]
        XtX = X_arr.T @ X_arr
        Xty = X_arr.T @ y_arr
        ridge_term = self.alpha * np.eye(n_features)
        try:
            beta = np.linalg.solve(XtX + ridge_term, Xty)
        except np.linalg.LinAlgError:
            logger.warning("Ridge regression solve failed (singular matrix)")
            return False

        # Intercept
        intercept = float(y_arr.mean() - beta @ X_z.mean().values)

        # R-squared score
        y_pred = X_arr @ beta + intercept
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Store results
        self.weights = {col: float(beta[i]) for i, col in enumerate(valid_cols)}
        self.intercept = intercept
        self.feature_means = {col: float(means[col]) for col in valid_cols}
        self.feature_stds = {col: float(stds[col]) for col in valid_cols}
        self.r_squared = float(r2)
        self.n_samples = len(common)

        logger.info("Linear factor trained: %d features, %d samples, R²=%.4f",
                     len(valid_cols), len(common), r2)
        return True

    def predict(self, signals: dict[str, float]) -> float:
        """Score a set of current signal values.

        Args:
            signals: Dict mapping signal name to numeric value.

        Returns:
            Predicted return (continuous). Positive = bullish, negative = bearish.
            Returns 0.0 if model not trained.
        """
        if not self.weights:
            return 0.0

        score = self.intercept
        for name, beta in self.weights.items():
            raw = signals.get(name, 0.0)
            mean = self.feature_means.get(name, 0.0)
            std = self.feature_stds.get(name, 1.0)
            z = (raw - mean) / std if std > 1e-10 else 0.0
            score += beta * z
        return float(score)

    def score_to_action(self, score: float, threshold: float = 0.001) -> tuple[str, float]:
        """Convert continuous score to BUY/SELL/HOLD action.

        Args:
            score: Predicted return from predict().
            threshold: Minimum absolute score for directional signal.

        Returns:
            (action, confidence) tuple.
        """
        if abs(score) < threshold:
            return "HOLD", 0.0
        action = "BUY" if score > 0 else "SELL"
        # Confidence proportional to score magnitude, capped at 0.8
        confidence = min(abs(score) / (threshold * 5), 0.8)
        return action, round(confidence, 4)

    def save(self, path: Path | None = None) -> None:
        """Persist model weights to JSON."""
        path = path or _MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights,
            "intercept": self.intercept,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "alpha": self.alpha,
            "r_squared": self.r_squared,
            "n_samples": self.n_samples,
        }
        atomic_write_json(path, data)

    def load(self, path: Path | None = None) -> bool:
        """Load model weights from JSON.

        Returns True if loaded successfully, False otherwise.
        """
        path = path or _MODEL_FILE
        data = load_json(path)
        if not data or "weights" not in data:
            return False
        self.weights = data["weights"]
        self.intercept = data.get("intercept", 0.0)
        self.feature_means = data.get("feature_means", {})
        self.feature_stds = data.get("feature_stds", {})
        self.alpha = data.get("alpha", 1.0)
        self.r_squared = data.get("r_squared", 0.0)
        self.n_samples = data.get("n_samples", 0)
        return True

    def feature_importance(self) -> list[tuple[str, float]]:
        """Return features sorted by absolute weight (most important first)."""
        return sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True)
