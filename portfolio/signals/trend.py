"""Composite trend signal — 7 sub-indicators with majority voting.

Sub-indicators:
    1. Golden Cross / Death Cross  (SMA50 vs SMA200 crossover)
    2. Moving Average Ribbon       (SMA 10/20/50/100/200 ordering)
    3. Price vs MA200              (close above/below SMA200)
    4. Supertrend(10, 3)           (ATR-based trend follower)
    5. Parabolic SAR(0.02, 0.2)    (trend direction via stop-and-reverse)
    6. Ichimoku Cloud              (price vs Senkou Span A/B)
    7. ADX(14) / +DI / -DI        (trend strength + direction)

Requires a DataFrame with columns: open, high, low, close, volume.
At least 200 rows recommended for SMA200; returns HOLD with degraded
confidence when insufficient data is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, rma, safe_float, sma, true_range

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_ROWS_FULL = 200       # ideal minimum for SMA200
_MIN_ROWS_BASIC = 30       # absolute minimum to attempt any calculation
_NUM_SUB_SIGNALS = 7


# ---------------------------------------------------------------------------
# Sub-signal 1: Golden Cross / Death Cross
# ---------------------------------------------------------------------------

def _golden_cross(close: pd.Series) -> str:
    """Detect SMA50/SMA200 crossover on the most recent bar.

    Returns BUY on golden cross, SELL on death cross, HOLD otherwise.
    """
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)

    if sma50.iloc[-1] is np.nan or sma200.iloc[-1] is np.nan:
        return "HOLD"
    if len(sma50.dropna()) < 2 or len(sma200.dropna()) < 2:
        return "HOLD"

    # Find the last two bars where both SMAs are valid
    valid = sma50.notna() & sma200.notna()
    if valid.sum() < 2:
        return "HOLD"

    curr_above = sma50.iloc[-1] > sma200.iloc[-1]
    prev_above = sma50.iloc[-2] > sma200.iloc[-2]

    if curr_above and not prev_above:
        return "BUY"
    if not curr_above and prev_above:
        return "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# Sub-signal 2: Moving Average Ribbon
# ---------------------------------------------------------------------------

def _ma_ribbon(close: pd.Series) -> str:
    """Check if SMA(10, 20, 50, 100, 200) are in bullish/bearish order."""
    periods = [10, 20, 50, 100, 200]
    values = []
    for p in periods:
        s = sma(close, p)
        val = s.iloc[-1]
        if pd.isna(val):
            return "HOLD"
        values.append(val)

    # Bullish: 10 > 20 > 50 > 100 > 200
    if all(values[i] > values[i + 1] for i in range(len(values) - 1)):
        return "BUY"
    # Bearish: 10 < 20 < 50 < 100 < 200
    if all(values[i] < values[i + 1] for i in range(len(values) - 1)):
        return "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# Sub-signal 3: Price vs MA200
# ---------------------------------------------------------------------------

def _price_vs_ma200(close: pd.Series) -> str:
    """Bullish if price > SMA200, bearish if below."""
    sma200 = sma(close, 200)
    val = sma200.iloc[-1]
    if pd.isna(val):
        return "HOLD"
    price = close.iloc[-1]
    if price > val:
        return "BUY"
    if price < val:
        return "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# Sub-signal 4: Supertrend(10, 3)
# ---------------------------------------------------------------------------

def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    """Classic Supertrend indicator.

    Returns:
        (supertrend_line, direction) where direction is +1 (uptrend/BUY)
        or -1 (downtrend/SELL).
    """
    tr = true_range(high, low, close)
    atr = rma(tr, period)

    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    n = len(close)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1, dtype=int)   # 1 = uptrend, -1 = downtrend

    close_arr = close.values
    upper_basic_arr = upper_basic.values
    lower_basic_arr = lower_basic.values

    for i in range(n):
        if np.isnan(upper_basic_arr[i]):
            continue

        # Upper band: never rises above previous upper band if previous close
        # was below it
        if i == 0 or np.isnan(upper_band[i - 1]):
            upper_band[i] = upper_basic_arr[i]
        else:
            if upper_basic_arr[i] < upper_band[i - 1] or close_arr[i - 1] > upper_band[i - 1]:
                upper_band[i] = upper_basic_arr[i]
            else:
                upper_band[i] = upper_band[i - 1]

        # Lower band: never falls below previous lower band if previous close
        # was above it
        if i == 0 or np.isnan(lower_band[i - 1]):
            lower_band[i] = lower_basic_arr[i]
        else:
            if lower_basic_arr[i] > lower_band[i - 1] or close_arr[i - 1] < lower_band[i - 1]:
                lower_band[i] = lower_basic_arr[i]
            else:
                lower_band[i] = lower_band[i - 1]

        # Direction & supertrend value
        if i == 0 or np.isnan(supertrend[i - 1]):
            direction[i] = 1
            supertrend[i] = lower_band[i]
        elif supertrend[i - 1] == upper_band[i - 1]:
            # Previously in downtrend
            if close_arr[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]
        else:
            # Previously in uptrend
            if close_arr[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]

    idx = close.index
    return pd.Series(supertrend, index=idx), pd.Series(direction, index=idx)


def _supertrend_signal(high: pd.Series, low: pd.Series,
                       close: pd.Series) -> tuple[str, float, int]:
    """Return (signal, supertrend_value, direction)."""
    st_line, st_dir = _supertrend(high, low, close)
    val = st_line.iloc[-1]
    d = st_dir.iloc[-1]
    if np.isnan(val):
        return "HOLD", float("nan"), 0
    if d == 1:
        return "BUY", float(val), int(d)
    return "SELL", float(val), int(d)


# ---------------------------------------------------------------------------
# Sub-signal 5: Parabolic SAR (0.02, 0.2)
# ---------------------------------------------------------------------------

def _parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                   af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Classic Parabolic SAR.

    Returns a Series of SAR values. SAR < close means uptrend,
    SAR > close means downtrend.
    """
    n = len(close)
    if n < 2:
        return pd.Series(np.full(n, np.nan), index=close.index)

    sar = np.full(n, np.nan)
    af = af_start
    is_long = True

    high_arr = high.values.astype(float)
    low_arr = low.values.astype(float)
    close_arr = close.values.astype(float)

    # Initialise: start in long if first close > previous close, else short
    if close_arr[1] >= close_arr[0]:
        is_long = True
        ep = high_arr[0]          # extreme point
        sar[0] = low_arr[0]
    else:
        is_long = False
        ep = low_arr[0]
        sar[0] = high_arr[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if np.isnan(prev_sar):
            sar[i] = prev_sar
            continue

        # Compute new SAR
        new_sar = prev_sar + af * (ep - prev_sar)

        if is_long:
            # SAR cannot be above the two previous lows
            if i >= 2:
                new_sar = min(new_sar, low_arr[i - 1], low_arr[i - 2])
            else:
                new_sar = min(new_sar, low_arr[i - 1])

            if low_arr[i] < new_sar:
                # Reversal to short
                is_long = False
                sar[i] = ep        # SAR flips to the extreme point
                ep = low_arr[i]
                af = af_start
            else:
                sar[i] = new_sar
                if high_arr[i] > ep:
                    ep = high_arr[i]
                    af = min(af + af_start, af_max)
        else:
            # SAR cannot be below the two previous highs
            if i >= 2:
                new_sar = max(new_sar, high_arr[i - 1], high_arr[i - 2])
            else:
                new_sar = max(new_sar, high_arr[i - 1])

            if high_arr[i] > new_sar:
                # Reversal to long
                is_long = True
                sar[i] = ep
                ep = high_arr[i]
                af = af_start
            else:
                sar[i] = new_sar
                if low_arr[i] < ep:
                    ep = low_arr[i]
                    af = min(af + af_start, af_max)

    return pd.Series(sar, index=close.index)


def _sar_signal(high: pd.Series, low: pd.Series,
                close: pd.Series) -> tuple[str, float]:
    """Return (signal, sar_value)."""
    sar = _parabolic_sar(high, low, close)
    val = sar.iloc[-1]
    if np.isnan(val):
        return "HOLD", float("nan")
    price = close.iloc[-1]
    if val < price:
        return "BUY", float(val)
    if val > price:
        return "SELL", float(val)
    return "HOLD", float(val)


# ---------------------------------------------------------------------------
# Sub-signal 6: Ichimoku Cloud
# ---------------------------------------------------------------------------

def _ichimoku_signal(high: pd.Series, low: pd.Series,
                     close: pd.Series) -> tuple[str, float, float]:
    """Ichimoku Cloud: price vs Senkou Span A/B.

    Tenkan-sen (9), Kijun-sen (26), Senkou Span A, Senkou Span B (52).
    Cloud is projected 26 periods ahead, but for a current-bar signal we
    compare the current close against the cloud value at the current bar
    (which was projected 26 bars ago).

    Returns (signal, tenkan, kijun).
    """
    def _midline(s: pd.Series, period: int) -> pd.Series:
        rh = s.rolling(window=period, min_periods=period).max()
        rl = s.rolling(window=period, min_periods=period).min()
        return (rh + rl) / 2.0

    # Need at least 52 + 26 = 78 bars for a meaningful cloud
    if len(close) < 78:
        return "HOLD", float("nan"), float("nan")

    tenkan = _midline(close, 9)       # Using close for simplicity (some use high/low)
    kijun = _midline(close, 26)

    # More accurate: use high/low for Tenkan and Kijun
    tenkan = (_rolling_high(high, 9) + _rolling_low(low, 9)) / 2.0
    kijun = (_rolling_high(high, 26) + _rolling_low(low, 26)) / 2.0

    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = ((_rolling_high(high, 52) + _rolling_low(low, 52)) / 2.0).shift(26)

    span_a = senkou_a.iloc[-1]
    span_b = senkou_b.iloc[-1]
    tenkan_val = tenkan.iloc[-1]
    kijun_val = kijun.iloc[-1]

    if pd.isna(span_a) or pd.isna(span_b):
        return "HOLD", safe_float(tenkan_val), safe_float(kijun_val)

    price = close.iloc[-1]
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    if price > cloud_top:
        signal = "BUY"
    elif price < cloud_bottom:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, safe_float(tenkan_val), safe_float(kijun_val)


def _rolling_high(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).max()


def _rolling_low(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).min()



# ---------------------------------------------------------------------------
# Sub-signal 7: ADX(14) / +DI / -DI
# ---------------------------------------------------------------------------

def _adx_di(high: pd.Series, low: pd.Series,
            close: pd.Series, period: int = 14) -> tuple[str, float, float, float]:
    """ADX with Directional Indicators.

    ADX > 25 and +DI > -DI = BUY (strong uptrend)
    ADX > 25 and -DI > +DI = SELL (strong downtrend)
    ADX < 20 = HOLD (no trend)
    20 <= ADX <= 25 = HOLD (weak/ambiguous trend)

    Returns (signal, adx, plus_di, minus_di).
    """
    n = len(close)
    if n < period + 1:
        return "HOLD", float("nan"), float("nan"), float("nan")

    high_arr = high.values.astype(float)
    low_arr = low.values.astype(float)

    # Directional movement
    up_move = np.diff(high_arr, prepend=np.nan)
    down_move = -np.diff(low_arr, prepend=np.nan)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    tr = true_range(high, low, close)

    # Wilder smoothing
    atr = rma(tr, period)
    smooth_plus_dm = rma(plus_dm_s, period)
    smooth_minus_dm = rma(minus_dm_s, period)

    plus_di = 100.0 * smooth_plus_dm / atr
    minus_di = 100.0 * smooth_minus_dm / atr

    # DX and ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * di_diff / di_sum.replace(0, np.nan)
    adx = rma(dx, period)

    adx_val = adx.iloc[-1]
    pdi_val = plus_di.iloc[-1]
    mdi_val = minus_di.iloc[-1]

    if pd.isna(adx_val) or pd.isna(pdi_val) or pd.isna(mdi_val):
        return "HOLD", float("nan"), float("nan"), float("nan")

    adx_val = float(adx_val)
    pdi_val = float(pdi_val)
    mdi_val = float(mdi_val)

    if adx_val > 25:
        if pdi_val > mdi_val:
            return "BUY", adx_val, pdi_val, mdi_val
        else:
            return "SELL", adx_val, pdi_val, mdi_val

    # ADX <= 25: no strong trend
    return "HOLD", adx_val, pdi_val, mdi_val


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_trend_signal(df: pd.DataFrame) -> dict:
    """Compute a composite trend signal from 7 sub-indicators.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV candle data with columns: open, high, low, close, volume.
        At least 200 rows recommended for full SMA200 coverage.
        Minimum ~30 rows required for any useful output.

    Returns
    -------
    dict
        {
            'action': 'BUY' | 'SELL' | 'HOLD',
            'confidence': float (0.0 - 1.0),
            'sub_signals': {
                'golden_cross': str,
                'ma_ribbon': str,
                'price_vs_ma200': str,
                'supertrend': str,
                'parabolic_sar': str,
                'ichimoku': str,
                'adx_di': str,
            },
            'indicators': {
                'sma50': float,
                'sma200': float,
                'supertrend': float,
                'supertrend_direction': int,
                'sar': float,
                'adx': float,
                'plus_di': float,
                'minus_di': float,
                'tenkan': float,
                'kijun': float,
            },
        }

    Notes
    -----
    - Returns HOLD with 0.0 confidence when data is insufficient.
    - Sub-indicators that lack enough data individually return HOLD.
    - Uses numpy/pandas only (no TA-Lib dependency).
    """
    # Default empty result
    default_result = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {k: "HOLD" for k in [
            "golden_cross", "ma_ribbon", "price_vs_ma200",
            "supertrend", "parabolic_sar", "ichimoku", "adx_di",
        ]},
        "indicators": {
            "sma50": float("nan"), "sma200": float("nan"),
            "supertrend": float("nan"), "supertrend_direction": 0,
            "sar": float("nan"),
            "adx": float("nan"), "plus_di": float("nan"), "minus_di": float("nan"),
            "tenkan": float("nan"), "kijun": float("nan"),
        },
    }

    # --- Validate input ------------------------------------------------
    if df is None or df.empty:
        return default_result

    required_cols = {"open", "high", "low", "close", "volume"}
    # Accept case-insensitive column names
    col_map = {c.lower(): c for c in df.columns}
    missing = required_cols - set(col_map.keys())
    if missing:
        return default_result

    # Normalise column access
    high = df[col_map["high"]].astype(float)
    low = df[col_map["low"]].astype(float)
    close = df[col_map["close"]].astype(float)

    if len(df) < _MIN_ROWS_BASIC:
        return default_result

    # --- Compute sub-signals ------------------------------------------

    # 1. Golden Cross / Death Cross
    gc_signal = _golden_cross(close)

    # 2. MA Ribbon
    ribbon_signal = _ma_ribbon(close)

    # 3. Price vs MA200
    pv200_signal = _price_vs_ma200(close)

    # 4. Supertrend
    st_signal, st_value, st_dir = _supertrend_signal(high, low, close)

    # 5. Parabolic SAR
    sar_signal, sar_value = _sar_signal(high, low, close)

    # 6. Ichimoku Cloud
    ichi_signal, tenkan_val, kijun_val = _ichimoku_signal(high, low, close)

    # 7. ADX / +DI / -DI
    adx_signal, adx_val, pdi_val, mdi_val = _adx_di(high, low, close)

    # --- Indicator values for output ----------------------------------
    sma50_val = sma(close, 50).iloc[-1]
    sma200_val = sma(close, 200).iloc[-1]

    indicators = {
        "sma50": safe_float(sma50_val),
        "sma200": safe_float(sma200_val),
        "supertrend": st_value,
        "supertrend_direction": st_dir,
        "sar": sar_value,
        "adx": adx_val,
        "plus_di": pdi_val,
        "minus_di": mdi_val,
        "tenkan": tenkan_val,
        "kijun": kijun_val,
    }

    # --- Majority vote ------------------------------------------------
    sub_signals_list = [
        gc_signal, ribbon_signal, pv200_signal,
        st_signal, sar_signal, ichi_signal, adx_signal,
    ]
    action, confidence = majority_vote(sub_signals_list)

    sub_signals = {
        "golden_cross": gc_signal,
        "ma_ribbon": ribbon_signal,
        "price_vs_ma200": pv200_signal,
        "supertrend": st_signal,
        "parabolic_sar": sar_signal,
        "ichimoku": ichi_signal,
        "adx_di": adx_signal,
    }

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
