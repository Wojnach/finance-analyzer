"""Composite advanced oscillators signal module.

Computes 8 oscillator sub-indicators and returns a majority-vote composite
BUY/SELL/HOLD signal with confidence score.

Sub-indicators:
    1. Awesome Oscillator          (AO zero-cross + twin-peaks saucer)
    2. Aroon Oscillator (25)       (trend strength via high/low recency)
    3. Vortex Indicator (14)       (VI+ vs VI- directional movement)
    4. Chande Momentum Osc (9)     (normalized momentum extremes)
    5. Know Sure Thing (KST)       (multi-ROC composite + signal crossover)
    6. Schaff Trend Cycle (23, 50) (MACD + double stochastic smoothing)
    7. TRIX (15)                   (triple-smoothed EMA rate of change)
    8. Coppock Curve (14, 11, 10)  (WMA of dual ROC, long-term buy signal)

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 50 rows of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Minimum rows required for reliable computation.  The longest lookback chain
# is the Coppock Curve (ROC-14 needs 14 bars + WMA-10 needs 10 more = 24)
# and Schaff Trend Cycle (50-period slow EMA warm-up), but we ask for 50
# to give every indicator a reasonable warm-up.
# ---------------------------------------------------------------------------
MIN_ROWS = 50


# ---- helpers ---------------------------------------------------------------

def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average using pandas ewm."""
    return series.ewm(span=span, adjust=False).mean()


def _wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted moving average (linearly weighted)."""
    weights = np.arange(1, period + 1, dtype=float)

    def _apply_wma(x: np.ndarray) -> float:
        return np.dot(x, weights) / weights.sum()

    return series.rolling(window=period, min_periods=period).apply(
        _apply_wma, raw=True,
    )


def _roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of change: 100 * (current - n_periods_ago) / n_periods_ago."""
    shifted = series.shift(period)
    return 100.0 * (series - shifted) / shifted.replace(0, np.nan)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _safe_float(val) -> float:
    """Convert to float, returning NaN for non-finite values."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return float("nan")
    try:
        f = float(val)
        return f if np.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ---- sub-indicator 1: Awesome Oscillator -----------------------------------

def _awesome_oscillator(high: pd.Series, low: pd.Series) -> tuple[float, str]:
    """Awesome Oscillator: SMA(5, median) - SMA(34, median).

    - AO crosses above 0 = BUY
    - AO crosses below 0 = SELL
    - Twin peaks (saucer): AO below 0, two dips where second is higher = BUY

    Returns (ao_value, signal).
    """
    median_price = (high + low) / 2.0
    ao = _sma(median_price, 5) - _sma(median_price, 34)

    val = ao.iloc[-1]
    if np.isnan(val):
        return float("nan"), "HOLD"

    # Need at least 2 values for crossover detection
    if len(ao.dropna()) < 2:
        return _safe_float(val), "HOLD"

    prev = ao.dropna().iloc[-2]
    if np.isnan(prev):
        return _safe_float(val), "HOLD"

    # Zero-line crossover
    if prev <= 0 and val > 0:
        return _safe_float(val), "BUY"
    if prev >= 0 and val < 0:
        return _safe_float(val), "SELL"

    # Twin peaks (saucer) detection: AO below zero, look for two local
    # minima in recent history where second dip is higher (less negative).
    # This is a bullish reversal signal.
    ao_clean = ao.dropna()
    if len(ao_clean) >= 10 and val < 0:
        recent = ao_clean.iloc[-10:].values
        # Find local minima (simple: value lower than both neighbours)
        minima_indices = []
        for i in range(1, len(recent) - 1):
            if recent[i] < recent[i - 1] and recent[i] < recent[i + 1]:
                minima_indices.append(i)

        if len(minima_indices) >= 2:
            first_dip = recent[minima_indices[-2]]
            second_dip = recent[minima_indices[-1]]
            # Second dip is higher (less negative) = bullish saucer
            if second_dip > first_dip and val > second_dip:
                return _safe_float(val), "BUY"

    return _safe_float(val), "HOLD"


# ---- sub-indicator 2: Aroon Oscillator (25) --------------------------------

def _aroon_oscillator(high: pd.Series, low: pd.Series,
                      period: int = 25) -> tuple[float, str]:
    """Aroon Oscillator = Aroon Up - Aroon Down.

    Aroon Up  = ((period - periods since period-high) / period) * 100
    Aroon Down = ((period - periods since period-low) / period) * 100
    Oscillator = Up - Down.

    Oscillator > 50 = BUY.  < -50 = SELL.

    Returns (aroon_osc_value, signal).
    """
    if len(high) < period + 1:
        return float("nan"), "HOLD"

    # Periods since period-high / period-low (0-based, 0 = current bar)
    high_window = high.iloc[-(period + 1):]
    low_window = low.iloc[-(period + 1):]

    periods_since_high = period - int(np.argmax(high_window.values))
    periods_since_low = period - int(np.argmin(low_window.values))

    aroon_up = ((period - periods_since_high) / period) * 100.0
    aroon_down = ((period - periods_since_low) / period) * 100.0
    osc = aroon_up - aroon_down

    if osc > 50:
        return osc, "BUY"
    if osc < -50:
        return osc, "SELL"
    return osc, "HOLD"


# ---- sub-indicator 3: Vortex Indicator (14) --------------------------------

def _vortex_indicator(high: pd.Series, low: pd.Series, close: pd.Series,
                      period: int = 14) -> tuple[float, float, str]:
    """Vortex Indicator.

    VM+ = |High_i - Low_{i-1}|
    VM- = |Low_i - High_{i-1}|
    Sum each over *period* bars, divide by sum of True Range.

    VI+ > VI- and VI+ > 1.0 = BUY.
    VI- > VI+ and VI- > 1.0 = SELL.

    Returns (vi_plus, vi_minus, signal).
    """
    if len(close) < period + 1:
        return float("nan"), float("nan"), "HOLD"

    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    tr = _true_range(high, low, close)

    vm_plus_sum = vm_plus.rolling(window=period, min_periods=period).sum()
    vm_minus_sum = vm_minus.rolling(window=period, min_periods=period).sum()
    tr_sum = tr.rolling(window=period, min_periods=period).sum()

    vi_plus = vm_plus_sum / tr_sum.replace(0, np.nan)
    vi_minus = vm_minus_sum / tr_sum.replace(0, np.nan)

    vip = vi_plus.iloc[-1]
    vim = vi_minus.iloc[-1]

    if np.isnan(vip) or np.isnan(vim):
        return float("nan"), float("nan"), "HOLD"

    vip = float(vip)
    vim = float(vim)

    if vip > vim and vip > 1.0:
        return vip, vim, "BUY"
    if vim > vip and vim > 1.0:
        return vip, vim, "SELL"
    return vip, vim, "HOLD"


# ---- sub-indicator 4: Chande Momentum Oscillator (9) -----------------------

def _chande_momentum(close: pd.Series, period: int = 9) -> tuple[float, str]:
    """Chande Momentum Oscillator.

    CMO = (sum_up - sum_down) / (sum_up + sum_down) * 100
    where sum_up / sum_down are sums of positive / negative price changes
    over *period* bars.

    CMO > 50 = BUY.  CMO < -50 = SELL.

    Returns (cmo_value, signal).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    sum_up = gain.rolling(window=period, min_periods=period).sum()
    sum_down = loss.rolling(window=period, min_periods=period).sum()

    denom = sum_up + sum_down
    cmo = 100.0 * (sum_up - sum_down) / denom.replace(0, np.nan)

    val = cmo.iloc[-1]
    if np.isnan(val):
        return float("nan"), "HOLD"

    val = float(val)
    if val > 50:
        return val, "BUY"
    if val < -50:
        return val, "SELL"
    return val, "HOLD"


# ---- sub-indicator 5: Know Sure Thing (KST) --------------------------------

def _know_sure_thing(close: pd.Series) -> tuple[float, float, str]:
    """Know Sure Thing.

    KST = SMA(10, ROC(10))*1 + SMA(10, ROC(15))*2
        + SMA(10, ROC(20))*3 + SMA(15, ROC(30))*4
    Signal = SMA(9, KST)

    KST crosses above signal = BUY.  Below signal = SELL.

    Returns (kst_value, kst_signal_value, signal).
    """
    # Standard KST ROC periods and their SMA smoothing periods + weights
    roc_periods = [10, 15, 20, 30]
    sma_periods = [10, 10, 10, 15]
    weights = [1, 2, 3, 4]

    # Need at least max(ROC period) + max(SMA period) bars
    min_needed = max(roc_periods) + max(sma_periods) + 1
    if len(close) < min_needed:
        return float("nan"), float("nan"), "HOLD"

    kst_line = pd.Series(0.0, index=close.index)
    for roc_p, sma_p, w in zip(roc_periods, sma_periods, weights):
        roc_val = _roc(close, roc_p)
        smoothed = _sma(roc_val, sma_p)
        kst_line = kst_line + w * smoothed

    signal_line = _sma(kst_line, 9)

    kst_val = kst_line.iloc[-1]
    sig_val = signal_line.iloc[-1]

    if np.isnan(kst_val) or np.isnan(sig_val):
        return _safe_float(kst_val), _safe_float(sig_val), "HOLD"

    # Need prior values for crossover detection
    if len(kst_line.dropna()) < 2 or len(signal_line.dropna()) < 2:
        return _safe_float(kst_val), _safe_float(sig_val), "HOLD"

    kst_prev = kst_line.dropna().iloc[-2]
    sig_prev = signal_line.dropna().iloc[-2]

    if np.isnan(kst_prev) or np.isnan(sig_prev):
        return _safe_float(kst_val), _safe_float(sig_val), "HOLD"

    # Bullish cross: KST crosses above signal
    if kst_prev <= sig_prev and kst_val > sig_val:
        return _safe_float(kst_val), _safe_float(sig_val), "BUY"

    # Bearish cross: KST crosses below signal
    if kst_prev >= sig_prev and kst_val < sig_val:
        return _safe_float(kst_val), _safe_float(sig_val), "SELL"

    return _safe_float(kst_val), _safe_float(sig_val), "HOLD"


# ---- sub-indicator 6: Schaff Trend Cycle (23, 50) --------------------------

def _schaff_trend_cycle(close: pd.Series, fast: int = 23,
                        slow: int = 50, cycle: int = 10,
                        ) -> tuple[float, str]:
    """Schaff Trend Cycle — MACD + double stochastic smoothing.

    1. Compute MACD line: EMA(fast) - EMA(slow)
    2. Apply stochastic transformation on MACD, smooth with EMA(cycle/2)
    3. Apply second stochastic transformation, smooth again

    STC > 75 = overbought (SELL).
    STC < 25 = oversold (BUY).
    STC crossing 50 upward = BUY.

    Returns (stc_value, signal).
    """
    if len(close) < slow + cycle:
        return float("nan"), "HOLD"

    # Step 1: MACD line
    macd = _ema(close, fast) - _ema(close, slow)

    # Step 2: First stochastic of MACD
    macd_low = macd.rolling(window=cycle, min_periods=cycle).min()
    macd_high = macd.rolling(window=cycle, min_periods=cycle).max()
    denom1 = macd_high - macd_low
    stoch1 = ((macd - macd_low) / denom1.replace(0, np.nan)) * 100.0
    # Smooth with EMA (half-cycle)
    pf = _ema(stoch1, max(cycle // 2, 1))

    # Step 3: Second stochastic of PF
    pf_low = pf.rolling(window=cycle, min_periods=cycle).min()
    pf_high = pf.rolling(window=cycle, min_periods=cycle).max()
    denom2 = pf_high - pf_low
    stoch2 = ((pf - pf_low) / denom2.replace(0, np.nan)) * 100.0
    # Smooth again
    stc = _ema(stoch2, max(cycle // 2, 1))

    val = stc.iloc[-1]
    if np.isnan(val):
        return float("nan"), "HOLD"

    val = float(val)

    # Overbought / oversold
    if val > 75:
        return val, "SELL"
    if val < 25:
        return val, "BUY"

    # STC crossing 50 upward
    stc_clean = stc.dropna()
    if len(stc_clean) >= 2:
        prev = float(stc_clean.iloc[-2])
        if not np.isnan(prev) and prev <= 50 and val > 50:
            return val, "BUY"

    return val, "HOLD"


# ---- sub-indicator 7: TRIX (15) --------------------------------------------

def _trix(close: pd.Series, period: int = 15,
          signal_period: int = 9) -> tuple[float, str]:
    """TRIX — triple-smoothed EMA, then rate of change.

    1. EMA1 = EMA(close, period)
    2. EMA2 = EMA(EMA1, period)
    3. EMA3 = EMA(EMA2, period)
    4. TRIX = 100 * (EMA3 - EMA3_prev) / EMA3_prev
    5. Signal = SMA(signal_period, TRIX)

    TRIX crosses above 0 = BUY.  Below 0 = SELL.
    TRIX crosses above signal line also valid as BUY.

    Returns (trix_value, signal).
    """
    ema1 = _ema(close, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)

    # Rate of change of triple-smoothed EMA (percentage)
    trix_line = 100.0 * (ema3 - ema3.shift(1)) / ema3.shift(1).replace(0, np.nan)
    signal_line = _sma(trix_line, signal_period)

    val = trix_line.iloc[-1]
    sig_val = signal_line.iloc[-1]

    if np.isnan(val):
        return float("nan"), "HOLD"

    val = float(val)

    # Need prior values for crossover detection
    trix_clean = trix_line.dropna()
    if len(trix_clean) < 2:
        return val, "HOLD"

    prev = float(trix_clean.iloc[-2])
    if np.isnan(prev):
        return val, "HOLD"

    # Zero-line crossover
    if prev <= 0 and val > 0:
        return val, "BUY"
    if prev >= 0 and val < 0:
        return val, "SELL"

    # Signal line crossover (secondary confirmation)
    if not np.isnan(sig_val):
        sig_clean = signal_line.dropna()
        if len(sig_clean) >= 2:
            sig_prev = float(sig_clean.iloc[-2])
            if not np.isnan(sig_prev):
                if prev <= sig_prev and val > float(sig_val):
                    return val, "BUY"
                if prev >= sig_prev and val < float(sig_val):
                    return val, "SELL"

    return val, "HOLD"


# ---- sub-indicator 8: Coppock Curve (14, 11, 10) ---------------------------

def _coppock_curve(close: pd.Series, roc_long: int = 14, roc_short: int = 11,
                   wma_period: int = 10) -> tuple[float, str]:
    """Coppock Curve = WMA(wma_period, ROC(roc_long) + ROC(roc_short)).

    Turns up from below 0 = BUY (classic long-term buy signal).
    No traditional sell signal; we generate SELL when CC turns down from
    above 0 for symmetry.

    Returns (coppock_value, signal).
    """
    min_needed = max(roc_long, roc_short) + wma_period + 1
    if len(close) < min_needed:
        return float("nan"), "HOLD"

    roc_l = _roc(close, roc_long)
    roc_s = _roc(close, roc_short)
    combined = roc_l + roc_s
    cc = _wma(combined, wma_period)

    val = cc.iloc[-1]
    if np.isnan(val):
        return float("nan"), "HOLD"

    val = float(val)

    cc_clean = cc.dropna()
    if len(cc_clean) < 2:
        return val, "HOLD"

    prev = float(cc_clean.iloc[-2])
    if np.isnan(prev):
        return val, "HOLD"

    # Classic Coppock BUY: curve turns up from below zero
    if val < 0 and val > prev:
        return val, "BUY"

    # Symmetric SELL: curve turns down from above zero
    if val > 0 and val < prev:
        return val, "SELL"

    return val, "HOLD"


# ---- majority vote ---------------------------------------------------------

def _majority_vote(signals: list[str]) -> tuple[str, float]:
    """Majority voting across sub-signals.

    Returns (action, confidence) where confidence is the proportion of
    sub-signals agreeing with the winning direction.
    """
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    hold_count = signals.count("HOLD")
    total = len(signals)

    if total == 0:
        return "HOLD", 0.0

    if buy_count > sell_count and buy_count > hold_count:
        return "BUY", round(buy_count / total, 4)
    if sell_count > buy_count and sell_count > hold_count:
        return "SELL", round(sell_count / total, 4)

    # Ties: if BUY == SELL (and both > HOLD), default to HOLD
    # If HOLD ties with either direction, default to HOLD
    if buy_count == sell_count and buy_count > hold_count:
        return "HOLD", round(hold_count / total, 4) if hold_count > 0 else 0.0

    return "HOLD", round(max(buy_count, sell_count, hold_count) / total, 4)


# ---- public API ------------------------------------------------------------

def compute_oscillator_signal(df: pd.DataFrame) -> dict:
    """Compute composite oscillator signal from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``open``, ``high``, ``low``, ``close``, ``volume``
        with at least 50 rows of numeric data.

    Returns
    -------
    dict
        ``action``       : ``'BUY'`` | ``'SELL'`` | ``'HOLD'``
        ``confidence``   : float 0.0-1.0 (proportion of sub-signals agreeing
                           with the majority action)
        ``sub_signals``  : per-indicator votes
        ``indicators``   : raw indicator values for downstream use
    """
    # -- Default / fallback result -----------------------------------------
    hold_result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "awesome": "HOLD",
            "aroon": "HOLD",
            "vortex": "HOLD",
            "chande": "HOLD",
            "kst": "HOLD",
            "schaff": "HOLD",
            "trix": "HOLD",
            "coppock": "HOLD",
        },
        "indicators": {
            "awesome_osc": float("nan"),
            "aroon_osc": float("nan"),
            "vi_plus": float("nan"),
            "vi_minus": float("nan"),
            "cmo": float("nan"),
            "kst": float("nan"),
            "kst_signal": float("nan"),
            "schaff": float("nan"),
            "trix": float("nan"),
            "coppock": float("nan"),
        },
    }

    # -- Input validation --------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        return hold_result

    required_cols = {"open", "high", "low", "close", "volume"}
    # Accept case-insensitive column names
    col_map = {c.lower(): c for c in df.columns}
    missing = required_cols - set(col_map.keys())
    if missing:
        return hold_result

    if len(df) < MIN_ROWS:
        return hold_result

    # Work on a clean copy to avoid mutating the caller's data
    df = df.copy()
    close = df[col_map["close"]].astype(float)
    high = df[col_map["high"]].astype(float)
    low = df[col_map["low"]].astype(float)

    # -- Compute each sub-indicator ----------------------------------------
    sub_signals: dict[str, str] = {}
    indicators: dict[str, float] = {}

    # 1. Awesome Oscillator
    try:
        ao_val, ao_sig = _awesome_oscillator(high, low)
        sub_signals["awesome"] = ao_sig
        indicators["awesome_osc"] = ao_val
    except Exception:
        sub_signals["awesome"] = "HOLD"
        indicators["awesome_osc"] = float("nan")

    # 2. Aroon Oscillator
    try:
        aroon_val, aroon_sig = _aroon_oscillator(high, low)
        sub_signals["aroon"] = aroon_sig
        indicators["aroon_osc"] = aroon_val
    except Exception:
        sub_signals["aroon"] = "HOLD"
        indicators["aroon_osc"] = float("nan")

    # 3. Vortex Indicator
    try:
        vip, vim, vortex_sig = _vortex_indicator(high, low, close)
        sub_signals["vortex"] = vortex_sig
        indicators["vi_plus"] = _safe_float(vip)
        indicators["vi_minus"] = _safe_float(vim)
    except Exception:
        sub_signals["vortex"] = "HOLD"
        indicators["vi_plus"] = float("nan")
        indicators["vi_minus"] = float("nan")

    # 4. Chande Momentum Oscillator
    try:
        cmo_val, cmo_sig = _chande_momentum(close)
        sub_signals["chande"] = cmo_sig
        indicators["cmo"] = _safe_float(cmo_val)
    except Exception:
        sub_signals["chande"] = "HOLD"
        indicators["cmo"] = float("nan")

    # 5. Know Sure Thing
    try:
        kst_val, kst_sig_val, kst_sig = _know_sure_thing(close)
        sub_signals["kst"] = kst_sig
        indicators["kst"] = kst_val
        indicators["kst_signal"] = kst_sig_val
    except Exception:
        sub_signals["kst"] = "HOLD"
        indicators["kst"] = float("nan")
        indicators["kst_signal"] = float("nan")

    # 6. Schaff Trend Cycle
    try:
        stc_val, stc_sig = _schaff_trend_cycle(close)
        sub_signals["schaff"] = stc_sig
        indicators["schaff"] = _safe_float(stc_val)
    except Exception:
        sub_signals["schaff"] = "HOLD"
        indicators["schaff"] = float("nan")

    # 7. TRIX
    try:
        trix_val, trix_sig = _trix(close)
        sub_signals["trix"] = trix_sig
        indicators["trix"] = _safe_float(trix_val)
    except Exception:
        sub_signals["trix"] = "HOLD"
        indicators["trix"] = float("nan")

    # 8. Coppock Curve
    try:
        cc_val, cc_sig = _coppock_curve(close)
        sub_signals["coppock"] = cc_sig
        indicators["coppock"] = _safe_float(cc_val)
    except Exception:
        sub_signals["coppock"] = "HOLD"
        indicators["coppock"] = float("nan")

    # -- Majority vote -----------------------------------------------------
    votes = list(sub_signals.values())
    action, confidence = _majority_vote(votes)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
