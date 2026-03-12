"""Technical indicators computed from kline data for the Elongir bot.

All indicator functions take lists of floats (close prices, volumes, etc.)
and return scalar values. The IndicatorSet bundles multi-timeframe indicators
for a single snapshot.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

from portfolio.elongir.data_provider import MarketSnapshot


# ---------------------------------------------------------------------------
# Core indicator functions
# ---------------------------------------------------------------------------

def compute_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Compute RSI from close prices. Returns None if insufficient data."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = deltas[-period:]
    gains = [d if d > 0 else 0.0 for d in recent]
    losses = [-d if d < 0 else 0.0 for d in recent]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute MACD line, signal line, and histogram.

    Returns (macd_line, signal_line, histogram) or (None, None, None).
    """
    if len(closes) < slow + signal_period:
        return None, None, None

    # Compute MACD line = EMA(fast) - EMA(slow) for each point
    fast_ema = _ema_series(closes, fast)
    slow_ema = _ema_series(closes, slow)

    if fast_ema is None or slow_ema is None:
        return None, None, None

    # Align lengths (slow_ema is shorter)
    offset = len(fast_ema) - len(slow_ema)
    macd_series = [
        fast_ema[offset + i] - slow_ema[i]
        for i in range(len(slow_ema))
    ]

    if len(macd_series) < signal_period:
        return None, None, None

    # Signal line = EMA of MACD series
    signal_ema = _ema_series(macd_series, signal_period)
    if signal_ema is None or len(signal_ema) == 0:
        return None, None, None

    macd_line = macd_series[-1]
    signal_line = signal_ema[-1]
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bb(
    closes: list[float],
    period: int = 20,
    std_mult: float = 2.0,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute Bollinger Bands (lower, mid, upper).

    Returns (lower, mid, upper) or (None, None, None).
    """
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return mid - std_mult * std, mid, mid + std_mult * std


def compute_ema(values: list[float], period: int) -> Optional[float]:
    """Compute current EMA value. Returns None if insufficient data."""
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1.0 - k)
    return ema


def compute_volume_ratio(
    volumes: list[float],
    recent: int = 5,
    avg_period: int = 20,
) -> Optional[float]:
    """Volume ratio: average of recent N bars / average of last avg_period bars.

    Returns None if insufficient data.
    """
    if len(volumes) < avg_period:
        return None
    avg_vol = sum(volumes[-avg_period:]) / avg_period
    if avg_vol <= 0:
        return None
    recent_vol = sum(volumes[-recent:]) / recent
    return recent_vol / avg_vol


def compute_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> Optional[float]:
    """Compute Average True Range. Returns None if insufficient data."""
    n = len(closes)
    if n < period + 1 or len(highs) < period + 1 or len(lows) < period + 1:
        return None
    trs = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    # Simple average of last `period` true ranges
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ema_series(values: list[float], period: int) -> Optional[list[float]]:
    """Compute a full EMA series from values."""
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema_val = sum(values[:period]) / period
    result = [ema_val]
    for v in values[period:]:
        ema_val = v * k + ema_val * (1.0 - k)
        result.append(ema_val)
    return result


def _extract_ohlcv(klines: list) -> Tuple[
    list[float], list[float], list[float], list[float], list[float]
]:
    """Extract OHLCV from raw Binance kline arrays.

    Kline format: [open_time, open, high, low, close, volume, ...]
    Returns (opens, highs, lows, closes, volumes).
    """
    opens = [float(k[1]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    closes = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    return opens, highs, lows, closes, volumes


# ---------------------------------------------------------------------------
# Timeframe indicators
# ---------------------------------------------------------------------------

@dataclass
class TimeframeIndicators:
    """Indicators for a single timeframe."""
    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_mid: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_position: Optional[str] = None  # "above_upper" | "below_lower" | "inside"
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    volume_ratio: Optional[float] = None
    atr: Optional[float] = None
    last_close: Optional[float] = None
    high_1h: Optional[float] = None     # highest close in the last ~60 min


def _compute_timeframe(klines: Optional[list]) -> TimeframeIndicators:
    """Compute indicators for a single timeframe from raw klines."""
    ti = TimeframeIndicators()
    if not klines or len(klines) < 2:
        return ti

    opens, highs, lows, closes, volumes = _extract_ohlcv(klines)
    ti.last_close = closes[-1] if closes else None

    ti.rsi = compute_rsi(closes)
    ml, sl, hist = compute_macd(closes)
    ti.macd_line = ml
    ti.macd_signal = sl
    ti.macd_histogram = hist

    bl, bm, bu = compute_bb(closes)
    ti.bb_lower = bl
    ti.bb_mid = bm
    ti.bb_upper = bu
    if bl is not None and bu is not None and closes:
        price = closes[-1]
        if price > bu:
            ti.bb_position = "above_upper"
        elif price < bl:
            ti.bb_position = "below_lower"
        else:
            ti.bb_position = "inside"

    ti.ema_9 = compute_ema(closes, 9)
    ti.ema_21 = compute_ema(closes, 21)
    ti.volume_ratio = compute_volume_ratio(volumes)
    ti.atr = compute_atr(highs, lows, closes)

    # For 1m klines: high in last 60 candles = 1h high
    # For 5m klines: high in last 12 candles = 1h high
    # For 15m klines: high in last 4 candles = 1h high
    if highs:
        ti.high_1h = max(highs[-60:]) if len(highs) >= 60 else max(highs)

    return ti


# ---------------------------------------------------------------------------
# IndicatorSet
# ---------------------------------------------------------------------------

@dataclass
class IndicatorSet:
    """Multi-timeframe indicator bundle."""
    tf_1m: TimeframeIndicators = field(default_factory=TimeframeIndicators)
    tf_5m: TimeframeIndicators = field(default_factory=TimeframeIndicators)
    tf_15m: TimeframeIndicators = field(default_factory=TimeframeIndicators)
    silver_usd: float = 0.0
    fx_rate: float = 0.0


def compute_all(snapshot: MarketSnapshot) -> IndicatorSet:
    """Compute indicators across all timeframes from a snapshot."""
    iset = IndicatorSet(
        silver_usd=snapshot.silver_usd,
        fx_rate=snapshot.fx_rate,
    )
    iset.tf_1m = _compute_timeframe(snapshot.klines_1m)
    iset.tf_5m = _compute_timeframe(snapshot.klines_5m)
    iset.tf_15m = _compute_timeframe(snapshot.klines_15m)

    # Adjust 1h high lookback for different timeframe resolutions
    # 5m klines: last 12 candles = 1h
    if snapshot.klines_5m and len(snapshot.klines_5m) >= 12:
        highs_5m = [float(k[2]) for k in snapshot.klines_5m]
        iset.tf_5m.high_1h = max(highs_5m[-12:])

    # 15m klines: last 4 candles = 1h
    if snapshot.klines_15m and len(snapshot.klines_15m) >= 4:
        highs_15m = [float(k[2]) for k in snapshot.klines_15m]
        iset.tf_15m.high_1h = max(highs_15m[-4:])

    return iset
