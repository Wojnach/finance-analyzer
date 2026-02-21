import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "user_data"
    / "data"
    / "binance"
    / "futures"
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
PAIRS = [
    ("BTC_USDT_USDT-1h-futures.feather", 0),
    ("ETH_USDT_USDT-1h-futures.feather", 1),
]


def compute_features(df, symbol_flag=0):
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    opn = df["open"].astype(float)
    volume = df["volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    eps = np.finfo(float).eps
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    feats["rsi14"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, eps)))

    # RSI(7)
    avg_gain7 = gain.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    avg_loss7 = loss.ewm(alpha=1 / 7, min_periods=7, adjust=False).mean()
    feats["rsi7"] = 100 - (100 / (1 + avg_gain7 / avg_loss7.replace(0, eps)))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    feats["macd_hist"] = macd_hist
    feats["macd_slope"] = macd_hist - macd_hist.shift(1)

    # EMA ratio & crossover
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    feats["ema_ratio"] = ema9 / ema21.replace(0, np.nan)
    feats["ema_cross"] = (ema9 > ema21).astype(int)

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    feats["bb_pctb"] = (close - bb_lower) / bb_range
    feats["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    # ATR%
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    feats["atr_pct"] = atr14 / close.replace(0, np.nan) * 100

    # Returns
    for n in [1, 3, 6, 12, 24]:
        feats[f"ret_{n}"] = close.pct_change(n)

    # Volume ratio
    feats["vol_ratio"] = volume / volume.rolling(20).mean().replace(0, np.nan)

    # Candle shape
    body_top = pd.concat([opn, close], axis=1).max(axis=1)
    body_bot = pd.concat([opn, close], axis=1).min(axis=1)
    close_safe = close.replace(0, np.nan)
    feats["upper_wick_pct"] = (high - body_top) / close_safe * 100
    feats["lower_wick_pct"] = (body_bot - low) / close_safe * 100
    feats["range_pct"] = (high - low) / close_safe * 100

    # Hour of day
    dt = pd.to_datetime(df["date"])
    hour = dt.dt.hour
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Symbol flag
    feats["symbol"] = symbol_flag

    return feats


def make_target(close):
    fwd = close.shift(-12) / close - 1
    target = pd.cut(fwd, bins=[-np.inf, -0.02, 0.02, np.inf], labels=[2, 0, 1])
    return target.astype(float)


def load_data():
    frames = []
    for fname, sym_flag in PAIRS:
        path = DATA_DIR / fname
        df = pd.read_feather(path)
        df = df.sort_values("date").reset_index(drop=True)
        feats = compute_features(df, symbol_flag=sym_flag)
        target = make_target(df["close"].astype(float))
        combined = feats.copy()
        combined["target"] = target
        combined["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
        frames.append(combined)
    return pd.concat(frames, ignore_index=True)


def walk_forward(data, feature_cols):
    months = sorted(data["month"].unique())
    all_preds = []
    all_true = []

    for i in range(6, len(months) - 1):
        train_months = months[: i + 1]
        test_month = months[i + 1]

        train = data[data["month"].isin(train_months)].dropna(
            subset=feature_cols + ["target"]
        )
        test = data[data["month"] == test_month].dropna(
            subset=feature_cols + ["target"]
        )

        if len(train) < 100 or len(test) < 10:
            continue

        X_train = train[feature_cols].values
        y_train = train["target"].astype(int).values
        X_test = test[feature_cols].values
        y_test = test["target"].astype(int).values

        model = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=50,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"  {test_month}: train={len(train):,} test={len(test):,} acc={acc:.3f}")

        all_preds.extend(preds)
        all_true.extend(y_test)

    print("\nOverall walk-forward classification report:")
    print(
        classification_report(all_true, all_preds, target_names=["HOLD", "BUY", "SELL"])
    )


def train_final(data, feature_cols):
    clean = data.dropna(subset=feature_cols + ["target"])
    X = clean[feature_cols].values
    y = clean["target"].astype(int).values

    model = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "ml_classifier.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "ml_feature_names.joblib")
    print(f"\nModel saved to {MODEL_DIR / 'ml_classifier.joblib'}")
    print(f"Feature names saved ({len(feature_cols)} features)")


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    feature_cols = [c for c in data.columns if c not in ("target", "month")]
    print(f"Dataset: {len(data):,} rows, {len(feature_cols)} features")
    print(f"Target distribution:\n{data['target'].value_counts().sort_index()}\n")

    print("Walk-forward validation:")
    walk_forward(data, feature_cols)

    print("Training final model on all data...")
    train_final(data, feature_cols)
