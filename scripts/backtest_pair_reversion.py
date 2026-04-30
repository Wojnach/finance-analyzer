"""Quick backtest of residual_pair_reversion signal."""
import json
import numpy as np
import pandas as pd
import yfinance as yf


def rolling_ols_beta(target_ret, driver_ret, window=180):
    cov = target_ret.rolling(window=window, min_periods=window).cov(driver_ret)
    var = driver_ret.rolling(window=window, min_periods=window).var()
    var_safe = var.replace(0, np.nan)
    beta = cov / var_safe
    residual = target_ret - beta * driver_ret
    return beta, residual


def compute_half_life(residual):
    resid = residual.dropna()
    if len(resid) < 30:
        return float("nan")
    y = resid.values[1:]
    x = resid.values[:-1]
    cov_xy = np.mean((x - np.mean(x)) * (y - np.mean(y)))
    var_x = np.var(x)
    if var_x == 0:
        return float("nan")
    theta = cov_xy / var_x
    if theta >= 1.0 or theta <= -1.0:
        return float("nan")
    if theta <= 0:
        theta = abs(theta)
    if theta >= 1.0:
        return float("nan")
    return -np.log(2) / np.log(abs(theta))


def backtest_pair(target_ticker, driver_ticker, name=None):
    name = name or target_ticker
    print(f"Backtesting {name} ({target_ticker} ~ {driver_ticker})...")

    target_data = yf.download(target_ticker, period="1y", interval="1d",
                              progress=False, auto_adjust=True)
    driver_data = yf.download(driver_ticker, period="1y", interval="1d",
                              progress=False, auto_adjust=True)

    if target_data is None or driver_data is None:
        print(f"  No data for {name}")
        return None
    if len(target_data) < 200 or len(driver_data) < 200:
        print(f"  Insufficient data: {len(target_data)}, {len(driver_data)}")
        return None

    target_close = target_data["Close"]
    driver_close = driver_data["Close"]
    if isinstance(target_close, pd.DataFrame):
        target_close = target_close.iloc[:, 0]
    if isinstance(driver_close, pd.DataFrame):
        driver_close = driver_close.iloc[:, 0]

    aligned = pd.DataFrame({"target": target_close, "driver": driver_close}).dropna()

    target_ret = np.log(aligned["target"] / aligned["target"].shift(1)).dropna()
    driver_ret = np.log(aligned["driver"] / aligned["driver"].shift(1)).dropna()
    common = target_ret.index.intersection(driver_ret.index)
    target_ret = target_ret.loc[common]
    driver_ret = driver_ret.loc[common]

    beta, residual = rolling_ols_beta(target_ret, driver_ret, window=180)

    correct_1d = correct_3d = correct_5d = 0
    total = 0

    for i in range(240, len(residual) - 5):
        res = residual.iloc[:i + 1].dropna()
        if len(res) < 60:
            continue

        mean = res.iloc[-60:].mean()
        std = res.iloc[-60:].std()
        if std == 0 or np.isnan(std):
            continue

        z = (res.iloc[-1] - mean) / std
        if abs(z) < 2.0:
            continue

        action = "BUY" if z < -2.0 else "SELL"
        total += 1

        idx = min(i + 1, len(aligned) - 1)
        future_1d = aligned["target"].iloc[idx] / aligned["target"].iloc[i] - 1
        idx3 = min(i + 3, len(aligned) - 1)
        future_3d = aligned["target"].iloc[idx3] / aligned["target"].iloc[i] - 1
        idx5 = min(i + 5, len(aligned) - 1)
        future_5d = aligned["target"].iloc[idx5] / aligned["target"].iloc[i] - 1

        if action == "BUY":
            if future_1d > 0.0005:
                correct_1d += 1
            if future_3d > 0.0005:
                correct_3d += 1
            if future_5d > 0.0005:
                correct_5d += 1
        else:
            if future_1d < -0.0005:
                correct_1d += 1
            if future_3d < -0.0005:
                correct_3d += 1
            if future_5d < -0.0005:
                correct_5d += 1

    if total > 0:
        hl = compute_half_life(residual.dropna().iloc[-60:])
        result = {
            "total_signals": total,
            "accuracy_1d": round(correct_1d / total, 4),
            "accuracy_3d": round(correct_3d / total, 4),
            "accuracy_5d": round(correct_5d / total, 4),
            "half_life": round(hl, 2) if not np.isnan(hl) else "NaN",
        }
        print(f"  {name}: {total} signals, "
              f"1d={correct_1d/total:.1%}, "
              f"3d={correct_3d/total:.1%}, "
              f"5d={correct_5d/total:.1%}, "
              f"HL={result['half_life']}")
        return result
    else:
        print(f"  {name}: 0 signals (z never exceeded threshold)")
        return {"total_signals": 0, "note": "no signals generated"}


if __name__ == "__main__":
    pairs = [
        ("ETH-USD", "BTC-USD", "ETH-USD"),
        ("BTC-USD", "ETH-USD", "BTC-USD"),
        ("GC=F", "SI=F", "XAU-USD"),
        ("SI=F", "GC=F", "XAG-USD"),
    ]

    results = {}
    for target, driver, name in pairs:
        r = backtest_pair(target, driver, name)
        if r is not None:
            results[name] = r

    print("\n=== BACKTEST RESULTS ===")
    print(json.dumps(results, indent=2))
