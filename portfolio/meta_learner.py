"""LightGBM meta-learner for signal consensus.

Replaces hand-tuned weighted voting with a trained model that learns
which signals matter for which tickers at which horizons.

Training data: signal_log.db (SQLite) — historical signal votes + price outcomes.
Features: 30 signal votes (BUY=1, SELL=-1, HOLD=0) + aggregate + context.
Target: binary classification (price up=1, down=0).

Resource usage: ~200MB RAM, 1 CPU thread, 20-30 seconds training.
"""

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio.meta_learner")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
SIGNAL_DB = DATA_DIR / "signal_log.db"

HORIZONS = ["3h", "1d", "3d", "5d"]

# BUG-147: Import canonical list from tickers instead of maintaining a copy.
from portfolio.tickers import SIGNAL_NAMES
import contextlib

VOTE_MAP = {"BUY": 1, "SELL": -1, "HOLD": 0}

# BUG-148: Module-level model cache to avoid deserializing on every predict() call.
# Keyed by horizon. Each entry: (model, mtime) — reloads if file is newer.
_model_cache: dict[str, tuple] = {}

CRYPTO = {"BTC-USD", "ETH-USD"}
METALS = {"XAU-USD", "XAG-USD"}

_MIN_CHANGE_PCT = 0.05  # Filter flat outcomes
_TEST_DAYS = 10  # Last N days for test set

# Tuned per-horizon configs (from grid search across 48 combinations)
_HORIZON_CONFIG = {
    "3h": {
        "dedup_block_hours": 12,
        "params": {
            "n_estimators": 150, "max_depth": 4, "num_leaves": 15,
            "learning_rate": 0.05, "min_child_samples": 40,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.5, "reg_lambda": 3.0,
        },
    },
    "1d": {
        "dedup_block_hours": 8,
        "params": {
            "n_estimators": 150, "max_depth": 4, "num_leaves": 15,
            "learning_rate": 0.05, "min_child_samples": 40,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.5, "reg_lambda": 3.0,
        },
    },
    "3d": {
        "dedup_block_hours": 4,
        "params": {
            "n_estimators": 150, "max_depth": 4, "num_leaves": 15,
            "learning_rate": 0.05, "min_child_samples": 40,
            "subsample": 0.7, "colsample_bytree": 0.7,
            "reg_alpha": 0.5, "reg_lambda": 3.0,
        },
    },
    "5d": {
        "dedup_block_hours": 12,
        "params": {
            "n_estimators": 80, "max_depth": 2, "num_leaves": 4,
            "learning_rate": 0.02, "min_child_samples": 80,
            "subsample": 0.5, "colsample_bytree": 0.5,
            "reg_alpha": 2.0, "reg_lambda": 10.0,
        },
    },
}


def _load_data(horizon="1d"):
    """Load training data from SQLite signal_log.db."""
    if not SIGNAL_DB.exists():
        raise FileNotFoundError(f"Signal database not found: {SIGNAL_DB}")

    conn = sqlite3.connect(str(SIGNAL_DB))
    try:
        query = """
            SELECT s.ts, ts.ticker, ts.signals, ts.regime,
                   o.change_pct
            FROM snapshots s
            JOIN ticker_signals ts ON ts.snapshot_id = s.id
            JOIN outcomes o ON o.snapshot_id = s.id AND o.ticker = ts.ticker
            WHERE o.horizon = ?
              AND o.change_pct IS NOT NULL
              AND ts.signals IS NOT NULL
        """
        df = pd.read_sql_query(query, conn, params=(horizon,))
    finally:
        conn.close()
    logger.info("Loaded %d raw rows for horizon=%s", len(df), horizon)
    return df


def _build_features(df):
    """Convert raw signal data to feature matrix."""
    records = []
    for _, row in df.iterrows():
        try:
            signals = json.loads(row["signals"]) if isinstance(row["signals"], str) else row["signals"]
        except (json.JSONDecodeError, TypeError):
            continue

        features = {}

        # Signal votes (30 features)
        for sig in SIGNAL_NAMES:
            features[f"sig_{sig}"] = VOTE_MAP.get(signals.get(sig, "HOLD"), 0)

        # Aggregate features
        n_buy = sum(1 for s in SIGNAL_NAMES if signals.get(s) == "BUY")
        n_sell = sum(1 for s in SIGNAL_NAMES if signals.get(s) == "SELL")
        n_hold = sum(1 for s in SIGNAL_NAMES if signals.get(s, "HOLD") == "HOLD")
        features["n_buy"] = n_buy
        features["n_sell"] = n_sell
        features["n_hold"] = n_hold
        features["buy_sell_ratio"] = n_buy / max(n_sell, 1)

        # Context features
        try:
            ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
            features["hour_utc"] = ts.hour
            features["day_of_week"] = ts.weekday()
        except (ValueError, AttributeError):
            features["hour_utc"] = 12
            features["day_of_week"] = 2

        ticker = row["ticker"]
        if ticker in CRYPTO:
            features["asset_class"] = 0
        elif ticker in METALS:
            features["asset_class"] = 1
        else:
            features["asset_class"] = 2

        features["change_pct"] = row["change_pct"]
        features["ts"] = row["ts"]
        features["ticker"] = ticker

        records.append(features)

    return pd.DataFrame(records)


def _deduplicate(df, block_hours=4):
    """Keep 1 entry per ticker per N-hour block to avoid temporal autocorrelation."""
    try:
        df["_ts"] = pd.to_datetime(df["ts"], utc=True)
    except Exception:
        df["_ts"] = pd.to_datetime(df["ts"])
    df["_block"] = df["_ts"].dt.floor(f"{block_hours}h")
    deduped = df.groupby(["ticker", "_block"]).first().reset_index()
    deduped = deduped.drop(columns=["_block", "_ts"], errors="ignore")
    logger.info("Deduplicated %d -> %d rows (%dh blocks)", len(df), len(deduped), block_hours)
    return deduped


def _prepare_xy(df):
    """Split into features X, target y, and metadata."""
    # Filter flat outcomes
    df = df[df["change_pct"].abs() >= _MIN_CHANGE_PCT].copy()

    # Target: 1 if price went up, 0 if down
    y = (df["change_pct"] > 0).astype(int)

    # Feature columns (everything except metadata and target)
    meta_cols = {"change_pct", "ts", "ticker"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols]

    return X, y, df[["ts", "ticker"]]


def train(horizon="1d", verbose=True):
    """Train a LightGBM meta-learner for the given horizon.

    Returns (model, metrics_dict).
    """
    import lightgbm as lgb

    # Load and prepare data
    raw = _load_data(horizon)
    if len(raw) < 100:
        raise ValueError(f"Insufficient data for horizon={horizon}: {len(raw)} rows")

    # Get tuned config for this horizon
    hcfg = _HORIZON_CONFIG.get(horizon, _HORIZON_CONFIG["1d"])
    block_hours = hcfg["dedup_block_hours"]
    tuned_params = hcfg["params"]

    features = _build_features(raw)
    features = _deduplicate(features, block_hours=block_hours)
    X, y, meta = _prepare_xy(features)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training LightGBM meta-learner for {horizon} horizon")
        print(f"{'='*60}")
        print(f"Samples after dedup+filter: {len(X)}")
        print(f"Class balance: {y.mean():.1%} UP / {1-y.mean():.1%} DOWN")
        print(f"Features: {len(X.columns)}")

    # Temporal train/test split
    try:
        ts_series = pd.to_datetime(meta["ts"], utc=True)
    except Exception:
        ts_series = pd.to_datetime(meta["ts"])
    cutoff = ts_series.max() - pd.Timedelta(days=_TEST_DAYS)
    train_mask = ts_series <= cutoff
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if verbose:
        print(f"Train: {len(X_train)} samples (up to {cutoff.date()})")
        print(f"Test:  {len(X_test)} samples (last {_TEST_DAYS} days)")

    # Categorical features
    cat_features = ["asset_class", "day_of_week"]

    # LightGBM parameters — tuned per horizon via grid search
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_threads": 1,          # Throttled: 1 core only
        "verbose": -1,             # Silent
        "random_state": 42,
        "is_unbalance": True,
        **tuned_params,
    }

    if verbose:
        print("\nTraining with num_threads=1, nice=19...")

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(0)],  # suppress per-round output
        categorical_feature=cat_features,
    )

    # Evaluate
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_proba)
    test_loss = log_loss(y_test, test_proba)

    metrics = {
        "horizon": horizon,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_auc": round(test_auc, 4),
        "test_logloss": round(test_loss, 4),
        "class_balance": round(y.mean(), 4),
        "features": len(X.columns),
        "trained_at": datetime.now(UTC).isoformat(),
    }

    if verbose:
        print(f"\n{'='*40}")
        print(f"Results ({horizon})")
        print(f"{'='*40}")
        print(f"Train accuracy: {train_acc:.1%}")
        print(f"Test accuracy:  {test_acc:.1%}")
        print(f"Test AUC:       {test_auc:.4f}")
        print(f"Test log-loss:  {test_loss:.4f}")

        # Feature importance (top 10)
        importance = model.feature_importances_
        feat_names = X.columns.tolist()
        top_idx = np.argsort(importance)[::-1][:10]
        print("\nTop 10 features:")
        for i, idx in enumerate(top_idx):
            print(f"  {i+1}. {feat_names[idx]:<25} importance={importance[idx]}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"meta_learner_{horizon}.joblib"
    joblib.dump(model, model_path)
    metrics_path = MODEL_DIR / f"meta_learner_{horizon}_metrics.json"
    from portfolio.file_utils import atomic_write_json
    atomic_write_json(metrics_path, metrics)

    if verbose:
        print(f"\nModel saved: {model_path}")
        print(f"Metrics saved: {metrics_path}")

    return model, metrics


def train_all(verbose=True):
    """Train meta-learners for all horizons."""
    results = {}
    for h in HORIZONS:
        try:
            _, metrics = train(h, verbose=verbose)
            results[h] = metrics
        except Exception as e:
            logger.error("Failed to train for horizon=%s: %s", h, e)
            if verbose:
                print(f"\nFailed for {h}: {e}")
            results[h] = {"error": str(e)}
    return results


def predict(votes, ticker, hour_utc=None, day_of_week=None, horizon="1d"):
    """Predict direction probability using trained meta-learner.

    Args:
        votes: dict mapping signal_name -> "BUY"/"SELL"/"HOLD"
        ticker: instrument ticker
        hour_utc: current hour (0-23), defaults to now
        day_of_week: 0=Monday, defaults to now
        horizon: prediction horizon

    Returns:
        (direction, probability) — "BUY"/"SELL"/"HOLD", float 0-1
    """
    model_path = MODEL_DIR / f"meta_learner_{horizon}.joblib"
    if not model_path.exists():
        return "HOLD", 0.0

    # BUG-148: Use cached model, reload only when file is newer (retrained).
    mtime = model_path.stat().st_mtime
    cached = _model_cache.get(horizon)
    if cached and cached[1] == mtime:
        model = cached[0]
    else:
        model = joblib.load(model_path)
        _model_cache[horizon] = (model, mtime)

    now = datetime.now(UTC)

    features = {}
    for sig in SIGNAL_NAMES:
        features[f"sig_{sig}"] = VOTE_MAP.get(votes.get(sig, "HOLD"), 0)

    n_buy = sum(1 for s in SIGNAL_NAMES if votes.get(s) == "BUY")
    n_sell = sum(1 for s in SIGNAL_NAMES if votes.get(s) == "SELL")
    features["n_buy"] = n_buy
    features["n_sell"] = n_sell
    features["n_hold"] = len(SIGNAL_NAMES) - n_buy - n_sell
    features["buy_sell_ratio"] = n_buy / max(n_sell, 1)
    features["hour_utc"] = hour_utc if hour_utc is not None else now.hour
    features["day_of_week"] = day_of_week if day_of_week is not None else now.weekday()

    if ticker in CRYPTO:
        features["asset_class"] = 0
    elif ticker in METALS:
        features["asset_class"] = 1
    else:
        features["asset_class"] = 2

    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0]  # [prob_down, prob_up]
    prob_up = proba[1]

    if prob_up > 0.55:
        return "BUY", round(prob_up, 4)
    elif prob_up < 0.45:
        return "SELL", round(1 - prob_up, 4)
    else:
        return "HOLD", round(max(prob_up, 1 - prob_up), 4)


if __name__ == "__main__":
    import os
    # Throttle: lowest priority
    with contextlib.suppress(OSError, AttributeError):
        os.nice(19)
    train_all(verbose=True)
